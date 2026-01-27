import os
import copy
import time
from datetime import datetime
from itertools import product  # Grid Search용
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
# [Regularization 실험 설정] 
# 가설: MHCA 모듈의 과적합을 막으면 성능이 오를 것이다.
# =============================================================================

# 1. 실험할 베이스 시나리오 (가장 검증하고 싶은 모델 하나를 선택)
# =============================================================================
# [Option 1] Early Fusion Only (DAEAC 스타일)
#   - ECG 1채널 + RR 2채널을 입력으로 결합
#   - Late Fusion 없음
# BASE_SCENARIO = {
#     "name": "1_Baseline_Lead3_EarlyFusion",
#     "config": {"fusion_type": None, "lead": 3, "rr_dim": 2},
#     "loader_key": "opt2"
# }
#
# [Option 2] Late Fusion Only (MHCA)
#   - ECG 1채널만 입력
#   - RR 7차원 특징을 MHCA로 Late Fusion
# BASE_SCENARIO = {
#     "name": "2_Clean_Lead1_MHCA",
#     "config": {"fusion_type": "mhca", "lead": 1, "rr_dim": 7, "fusion_num_heads": 1},
#     "loader_key": "opt1"
# }
#
# [Option 3] Early + Late Fusion (Redundant)
#   - ECG 1채널 + RR 2채널 입력 (Early Fusion)
#   - RR 7차원 특징을 MHCA로 Late Fusion
# BASE_SCENARIO = {
#     "name": "3_Redundant_Lead3_MHCA",
#     "config": {"fusion_type": "mhca", "lead": 3, "rr_dim": 7, "fusion_num_heads": 1},
#     "loader_key": "opt3"
# }
# =============================================================================

# 현재 선택: Early Fusion Only
BASE_SCENARIO = {
    "name": "1_Baseline_Lead3_EarlyFusion",
    "config": {"fusion_type": None, "lead": 3, "rr_dim": 2},
    "loader_key": "opt2"
}

# 2. 탐색할 하이퍼파라미터 그리드 (L2 & Dropout)
# - weight_decay (L2): 1e-4(약함) ~ 1e-2(강함)
# - dropout: 0.0(없음) ~ 0.3(적당함)
REGULARIZATION_GRID = {
    "weight_decay": [1e-4, 1e-3, 1e-2], 
    "dropout": [0.0, 0.3]
}

EXP_EPOCHS = 50 
OUTPUT_DIR = "./regularization_exp_202602/"

# =============================================================================
# 실행 로직
# =============================================================================

def run_single_experiment(base_config: dict, reg_params: dict, exp_name: str, 
                          loader_key: str, data_loaders: dict, device):
    """단일 실험 수행 (Regularization 파라미터 적용)"""
    
    loaders = data_loaders[loader_key]
    train_loader, valid_loader, test_loader = loaders

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Base Config: {base_config}")
    print(f"Regularization: {reg_params}")
    print(f"{'='*60}")

    set_seed(config.SEED)
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. 모델 설정 업데이트 (Dropout 적용)
    model_config = copy.deepcopy(config.MODEL_CONFIG)
    model_config.update(base_config)
    model_config['dropout'] = reg_params['dropout']  # Dropout 적용

    # 2. Optimizer 설정 업데이트 (Weight Decay / L2 적용)
    optimizer_config = copy.deepcopy(config.OPTIMIZER_CONFIG)
    optimizer_config['weight_decay'] = reg_params['weight_decay'] # L2 적용

    # 모델 생성
    ModelClass = MODEL_REGISTRY.get(config.MODEL_NAME)
    model = ModelClass(**model_config)

    # Loss 설정
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
    
    # Optimizer 생성 시 업데이트된 config 사용
    optimizer = build_optimizer(model.parameters(), config.OPTIMIZER_NAME, **optimizer_config)
    scheduler = build_scheduler(optimizer, config.SCHEDULER_NAME, **config.SCHEDULER_CONFIG)

    trainer = Trainer(
        model=model, train_loader=train_loader, valid_loader=valid_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, exp_dir=exp_dir, class_names=config.CLASSES,
    )

    trainer.fit(epochs=EXP_EPOCHS)

    # 결과 평가
    results = {'exp_name': exp_name, 'reg_params': reg_params, 'status': 'success', 'full_metrics': {}}
    for metric in ["macro_auprc", "macro_auroc", "macro_f1"]:
        if trainer.load_best_model(metric):
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

    # [1/3] 데이터 로드 (한 번만 수행)
    print("\n[1/3] Loading Data...")
    
    def load_split(records, split_name, extraction_style):
        return load_or_extract_data(records, config.DATA_PATH, config.VALID_LEADS, config.SIGNAL_LENGTH, split_name, extraction_style)

    data_loaders = {}

    # opt1 (Standard style) - Late Fusion Only
    train_data_opt1 = load_split(config.DS1_TRAIN, "Train", "default")
    valid_data_opt1 = load_split(config.DS1_VALID, "Valid", "default")
    test_data_opt1 = load_split(config.DS2_TEST, "Test", "default")

    DatasetClass_opt1 = DATASET_REGISTRY.get("ecg_standard")
    data_loaders["opt1"] = (
        DataLoader(DatasetClass_opt1(*train_data_opt1), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(DatasetClass_opt1(*valid_data_opt1), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(DatasetClass_opt1(*test_data_opt1), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    )

    # opt2 (DAEAC style) - Early Fusion Only
    train_data_opt2 = load_split(config.DS1_TRAIN, "Train", "daeac")
    valid_data_opt2 = load_split(config.DS1_VALID, "Valid", "daeac")
    test_data_opt2 = load_split(config.DS2_TEST, "Test", "daeac")

    DatasetClass_opt2 = DATASET_REGISTRY.get("daeac")
    data_loaders["opt2"] = (
        DataLoader(DatasetClass_opt2(*train_data_opt2), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(DatasetClass_opt2(*valid_data_opt2), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(DatasetClass_opt2(*test_data_opt2), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    )

    # [2/3] 실험 리스트 생성 (Cartesian Product)
    experiments = []
    keys, values = zip(*REGULARIZATION_GRID.items())
    for v in product(*values):
        reg_params = dict(zip(keys, v)) # {'weight_decay': 0.001, 'dropout': 0.0}
        
        # 실험 이름 자동 생성 (BASE_SCENARIO 이름 + regularization 파라미터)
        wd_str = f"{reg_params['weight_decay']:.0e}".replace('-', '')
        drop_str = f"{reg_params['dropout']}"
        exp_name = f"{BASE_SCENARIO['name']}_wd{wd_str}_drop{drop_str}"
        
        experiments.append((exp_name, reg_params))

    # [3/3] 실험 실행
    print(f"\n[3/3] Running {len(experiments)} Regularization Experiments...")
    
    # 템플릿 파일 확인
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "model_fusion.xlsx")
    excel_writer = None

    if os.path.exists(template_path):
        excel_path = os.path.join(OUTPUT_DIR, "results.xlsx")
        excel_writer = ExcelResultWriter(template_path, excel_path, config.CLASSES)
    else:
        print(f"Warning: Template file not found at {template_path}. Excel results will not be saved.")
    
    for exp_name, reg_params in experiments:
        try:
            res = run_single_experiment(
                BASE_SCENARIO["config"], 
                reg_params, 
                exp_name, 
                BASE_SCENARIO["loader_key"], 
                data_loaders, 
                device
            )
            
            # 결과 기록
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