import os
import copy
import time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.registry import MODEL_REGISTRY, LOSS_REGISTRY, DATASET_REGISTRY
from src.trainer import Trainer
from src.optimizers import build_optimizer
from src.schedulers import build_scheduler
from src.utils import ExcelResultWriter, CumulativeExcelWriter
from utils import set_seed, load_or_extract_data

# =============================================================================
# [검증 실험 설정] 가설: Backbone의 Early Fusion이 MHCA와 충돌하는가?
# =============================================================================

# 실험할 파라미터 직접 정의 (Grid Search가 아닌 시나리오 기반)
# 각 튜플은 (실험이름, 설정Dict, 사용할_데이터로더_키)
EXPERIMENT_SCENARIOS = [
    # 1. Baseline: Wang et al. (Lead 3, No Fusion)
    (
        "1_Baseline_Lead3_NoFusion",
        {"fusion_type": None, "lead": 3, "rr_dim": 2},
        "opt2" # DAEAC 데이터 로더 (Lead 3)
    ),

    # 2. Clean Late Fusion: ECG Only + MHCA (Lead 1, MHCA) -> 가설 검증의 핵심
    (
        "2_Clean_Lead1_MHCA",
        {"fusion_type": "mhca", "lead": 1, "rr_dim": 7, "fusion_num_heads": 1},
        "opt1" # Standard 데이터 로더 (Lead 1)
    ),

    # 3. Redundant Fusion: ECG+RR + MHCA (Lead 3, MHCA) -> 성능 하락의 원인 추정
    (
        "3_Redundant_Lead3_MHCA",
        {"fusion_type": "mhca", "lead": 3, "rr_dim": 7, "fusion_num_heads": 1},
        "opt3" # Early+Late Fusion 데이터 로더
    ),
]

EXP_EPOCHS = 50  # 비교를 위해 충분한 Epoch 필요
OUTPUT_DIR = "./verification_exp_202602/"

# =============================================================================
# 실행 로직 (기존 코드 재사용 및 단순화)
# =============================================================================

def run_single_experiment(exp_config: dict, exp_name: str, loader_key: str, data_loaders: dict, device):
    """단일 실험 수행"""
    # 데이터로더 선택
    loaders = data_loaders[loader_key]
    train_loader, valid_loader, test_loader = loaders

    print(f"\n{'='*60}")
    print(f"Scenario: {exp_name}")
    print(f"Config: {exp_config}")
    print(f"Loader: {loader_key}")
    print(f"{'='*60}")

    set_seed(config.SEED)
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 모델 설정 병합
    model_config = copy.deepcopy(config.MODEL_CONFIG)
    model_config.update(exp_config)

    # 모델 생성
    ModelClass = MODEL_REGISTRY.get(config.MODEL_NAME)
    model = ModelClass(**model_config)

    # Loss, Optimizer, Scheduler 설정 (기존과 동일)
    class_weights = None
    if config.LOSS_CONFIG.get('use_class_weights', False):
        labels = train_loader.dataset.labels
        if torch.is_tensor(labels): labels = labels.cpu().numpy()
        counts = np.bincount(labels.flatten().astype(np.int64), minlength=config.NUM_CLASSES)
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * config.NUM_CLASSES
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    LossClass = LOSS_REGISTRY.get(config.LOSS_NAME)
    criterion = LossClass(weight=class_weights)
    
    optimizer = build_optimizer(model.parameters(), config.OPTIMIZER_NAME, **config.OPTIMIZER_CONFIG)
    scheduler = build_scheduler(optimizer, config.SCHEDULER_NAME, **config.SCHEDULER_CONFIG)

    trainer = Trainer(
        model=model, train_loader=train_loader, valid_loader=valid_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, exp_dir=exp_dir, class_names=config.CLASSES,
    )

    trainer.fit(epochs=EXP_EPOCHS)

    # 결과 평가
    results = {'exp_name': exp_name, 'config': exp_config, 'status': 'success', 'full_metrics': {}}
    for metric in ["macro_auprc", "macro_auroc", "macro_f1"]: # F1도 중요 모니터링
        if trainer.load_best_model(metric): # 모델 로드 성공 시에만
            test_metrics = trainer.evaluate(test_loader)
            results[f"test_{metric}"] = test_metrics
            results['full_metrics'][metric] = test_metrics
            
    return results

def main():
    device = config.get_device()
    print(f"Device: {device}")
    
    global OUTPUT_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # [데이터 로드 섹션 - 기존 코드와 동일하게 opt1, opt2, opt3 모두 준비]
    print("\n[1/2] Loading Data...")
    
    # 1. Load opt2 (DAEAC style)
    def load_split(records, split_name, extraction_style):
        return load_or_extract_data(records, config.DATA_PATH, config.VALID_LEADS, config.SIGNAL_LENGTH, split_name, extraction_style)

    train_data_opt2 = load_split(config.DS1_TRAIN, "Train", "daeac")
    valid_data_opt2 = load_split(config.DS1_VALID, "Valid", "daeac")
    test_data_opt2 = load_split(config.DS2_TEST, "Test", "daeac")
    
    DatasetClass_opt2 = DATASET_REGISTRY.get("daeac")
    data_loaders = {}
    data_loaders["opt2"] = (
        DataLoader(DatasetClass_opt2(*train_data_opt2), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(DatasetClass_opt2(*valid_data_opt2), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(DatasetClass_opt2(*test_data_opt2), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    )

    # 2. Load opt1 (Standard style)
    train_data_opt1 = load_split(config.DS1_TRAIN, "Train", "default")
    valid_data_opt1 = load_split(config.DS1_VALID, "Valid", "default")
    test_data_opt1 = load_split(config.DS2_TEST, "Test", "default")
    
    DatasetClass_opt1 = DATASET_REGISTRY.get("ecg_standard")
    data_loaders["opt1"] = (
        DataLoader(DatasetClass_opt1(*train_data_opt1), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(DatasetClass_opt1(*valid_data_opt1), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(DatasetClass_opt1(*test_data_opt1), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    )

    # 3. Create opt3 (Matching)
    # (opt3 생성 로직은 기존 코드와 완전히 동일하므로 생략하지 않고 그대로 둠)
    def match_rr_features_7d(opt2_data, opt1_data):
        opt2_pids, opt2_sids = opt2_data[3], opt2_data[4]
        opt1_pids, opt1_sids, opt1_rr = opt1_data[3], opt1_data[4], opt1_data[2]
        opt1_map = { (int(p), int(s) if isinstance(s, (int, np.integer)) else s): i for i, (p, s) in enumerate(zip(opt1_pids, opt1_sids)) }
        
        matched_idx, matched_rr = [], []
        for i, (p, s) in enumerate(zip(opt2_pids, opt2_sids)):
            key = (int(p), int(s) if isinstance(s, (int, np.integer)) else s)
            if key in opt1_map:
                matched_idx.append(i)
                matched_rr.append(opt1_rr[opt1_map[key]])
        
        matched_data = opt2_data[0][matched_idx]
        matched_labels = opt2_data[1][matched_idx]
        matched_rr_2d = opt2_data[2][matched_idx]
        matched_pids = opt2_data[3][matched_idx]
        matched_sids = opt2_data[4][matched_idx]
        return (matched_data, matched_labels, matched_rr_2d, np.array(matched_rr), matched_pids, matched_sids)

    DatasetClass_opt3 = DATASET_REGISTRY.get("opt3")
    data_loaders["opt3"] = (
        DataLoader(DatasetClass_opt3(*match_rr_features_7d(train_data_opt2, train_data_opt1)), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS),
        DataLoader(DatasetClass_opt3(*match_rr_features_7d(valid_data_opt2, valid_data_opt1)), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS),
        DataLoader(DatasetClass_opt3(*match_rr_features_7d(test_data_opt2, test_data_opt1)), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    )

    # [실험 실행 섹션]
    print(f"\n[2/2] Running {len(EXPERIMENT_SCENARIOS)} Verification Scenarios...")
    
    # 엑셀 writer 초기화 (템플릿 파일 존재 확인)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "model_fusion.xlsx")
    excel_writer = None
    
    if os.path.exists(template_path):
        excel_path = os.path.join(OUTPUT_DIR, "results.xlsx")
        excel_writer = ExcelResultWriter(template_path, excel_path, config.CLASSES)
    else:
        print(f"Warning: Template file not found at {template_path}. Excel results will not be saved.")
    
    for exp_name, exp_config, loader_key in EXPERIMENT_SCENARIOS:
        try:
            res = run_single_experiment(exp_config, exp_name, loader_key, data_loaders, device)
            
            # 결과 기록 (Best AUPRC 기준)
            if 'test_macro_auprc' in res:
                metrics = res['full_metrics']['macro_auprc']
                if excel_writer:
                    excel_writer.write_metrics(exp_name, metrics, "auprc")
                print(f" -> {exp_name}: Acc={metrics['acc']:.4f}, F1={metrics['macro_f1']:.4f}, AUPRC={metrics['macro_auprc']:.4f}")
        except Exception as e:
            print(f"ERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()