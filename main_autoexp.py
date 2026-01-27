"""
Automated Experiment Script

여러 설정을 자동으로 실험합니다.
config.py의 설정을 동적으로 변경하면서 실험을 수행합니다.

사용법:
    python main_autoexp.py
"""

import os
import copy
import time
from datetime import datetime
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.registry import MODEL_REGISTRY, LOSS_REGISTRY, DATASET_REGISTRY
from src import models, losses, datasets
from src.optimizers import build_optimizer
from src.schedulers import build_scheduler
from src.trainer import Trainer
from src.utils import ExcelResultWriter, CumulativeExcelWriter
from utils import set_seed, load_or_extract_data


# =============================================================================
# 실험 설정 - 여기만 수정하면 됩니다
# =============================================================================

# 실험할 파라미터 그리드
EXPERIMENT_GRID = {
    # Fusion type 실험 (opt1, lead=1일 때만 의미있음)
    # "fusion_type": ["mhca"],
    "fusion_type": [None],

    # MHCA용 num_heads (fusion_type="mhca"일 때만 사용)
    "fusion_num_heads": [1],
}

# Fusion 실험용 설정 (late fusion은 opt1 스타일 필요)
FUSION_EXP_CONFIG = {
    "lead": 1,           # ECG only (RR은 late fusion으로)
    "rr_dim": 7,         # opt1: 7 features
    "dataset_name": "ecg_standard",
    "rr_feature_option": "opt1",
}


# 실험별 epochs (빠른 실험용)
EXP_EPOCHS = 50

# 결과 저장 경로
OUTPUT_DIR = "./autoexp_results_202602/"


# =============================================================================
# 실험 실행 함수
# =============================================================================

def run_single_experiment(exp_config: dict, exp_name: str, data_loaders: dict, device):
    """
    단일 실험 수행

    Args:
        exp_config: 실험 설정 dict (MODEL_CONFIG에 병합됨)
        exp_name: 실험 이름
        data_loaders: {"opt1": loaders, "opt2": loaders, "opt3": loaders} 형태
        device: torch device

    Returns:
        결과 dict
    """
    load_start = time.time()
    
    # 데이터로더 선택
    fusion_type = exp_config.get('fusion_type')
    is_opt3 = exp_config.get('use_opt3', False)
    
    if is_opt3:
        loaders = data_loaders["opt3"]
    elif fusion_type is not None and fusion_type.lower() != 'none':
        loaders = data_loaders["opt1"]
    else:
        loaders = data_loaders["opt2"]
    
    load_time = time.time() - load_start
    if load_time > 0.1:
        print(f"  [Load time: {load_time:.2f}s]")

    train_loader, valid_loader, test_loader = loaders

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Config: {exp_config}")
    print(f"{'='*60}")

    set_seed(config.SEED)

    # 실험 디렉토리
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 모델 설정 병합
    model_config = copy.deepcopy(config.MODEL_CONFIG)

    # opt3는 lead=3 (early fusion) + fusion_type (late fusion) + rr_dim=7
    if is_opt3:
        # opt3는 이미 exp_config에 설정되어 있음
        pass
    # Fusion 실험이면 opt1 설정 적용 (lead=1, rr_dim=7)
    elif fusion_type is not None and fusion_type.lower() != 'none':
        model_config.update({
            "lead": FUSION_EXP_CONFIG["lead"],
            "rr_dim": FUSION_EXP_CONFIG["rr_dim"],
        })

    model_config.update(exp_config)

    # 모델 생성
    ModelClass = MODEL_REGISTRY.get(config.MODEL_NAME)
    model = ModelClass(**model_config)

    total_params, _ = model.count_parameters()
    print(f"Model parameters: {total_params:,}")

    # Loss
    class_weights = None
    if config.LOSS_CONFIG.get('use_class_weights', False):
        labels = train_loader.dataset.labels
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        counts = np.bincount(labels.flatten().astype(np.int64), minlength=config.NUM_CLASSES)
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * config.NUM_CLASSES
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    LossClass = LOSS_REGISTRY.get(config.LOSS_NAME)
    criterion = LossClass(weight=class_weights)

    # Optimizer & Scheduler
    optimizer = build_optimizer(model.parameters(), config.OPTIMIZER_NAME, **config.OPTIMIZER_CONFIG)
    scheduler = build_scheduler(optimizer, config.SCHEDULER_NAME, **config.SCHEDULER_CONFIG)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        exp_dir=exp_dir,
        class_names=config.CLASSES,
    )

    # 학습
    trainer.fit(epochs=EXP_EPOCHS)

    # 테스트
    results = {
        'exp_name': exp_name,
        'config': exp_config,
        'status': 'success',
        'full_metrics': {},  # 엑셀 저장용 전체 metrics
    }

    for metric in ["macro_auprc", "macro_auroc", "macro_recall"]:
        trainer.load_best_model(metric)
        test_metrics = trainer.evaluate(test_loader)
        results[f"test_{metric}"] = {
            'acc': test_metrics['acc'],
            'macro_f1': test_metrics['macro_f1'],
            'macro_auprc': test_metrics['macro_auprc'],
            'macro_auroc': test_metrics['macro_auroc'],
        }
        results['full_metrics'][metric] = test_metrics  # 전체 metrics 저장

    return results


def generate_experiments():
    """실험 설정 조합 생성"""
    experiments = []

    for fusion_type in EXPERIMENT_GRID["fusion_type"]:
        if fusion_type is None:
            # Baseline: opt2 (DAEAC style, early fusion)
            exp_config = {
                "fusion_type": None,
                "lead": 3,      # opt2: ECG + 2 RR channels
                "rr_dim": 2,    # opt2: 2 RR features
            }
            exp_name = "baseline_opt2"
            experiments.append((exp_config, exp_name))
        elif fusion_type == "mhca":
            # MHCA는 num_heads별로 실험
            # opt1 style (late fusion only)
            for num_heads in EXPERIMENT_GRID["fusion_num_heads"]:
                exp_config = {
                    "fusion_type": fusion_type,
                    "fusion_num_heads": num_heads,
                }
                exp_name = f"mhca_h{num_heads}"
                experiments.append((exp_config, exp_name))
            
            # opt3 style (early + late fusion)
            for num_heads in EXPERIMENT_GRID["fusion_num_heads"]:
                exp_config = {
                    "fusion_type": fusion_type,
                    "fusion_num_heads": num_heads,
                    "lead": 3,      # opt3: ECG + 2 RR channels (early fusion)
                    "rr_dim": 7,    # opt3: 7 RR features (late fusion)
                    "use_opt3": True,
                }
                exp_name = f"opt3_mhca_h{num_heads}"
                experiments.append((exp_config, exp_name))
        else:
            # concat, concat_proj: opt1 style (late fusion only)
            exp_config = {"fusion_type": fusion_type}
            exp_name = f"fusion_{fusion_type}"
            experiments.append((exp_config, exp_name))
            
            # opt3 style (early + late fusion)
            exp_config = {
                "fusion_type": fusion_type,
                "lead": 3,      # opt3: ECG + 2 RR channels (early fusion)
                "rr_dim": 7,    # opt3: 7 RR features (late fusion)
                "use_opt3": True,
            }
            exp_name = f"opt3_{fusion_type}"
            experiments.append((exp_config, exp_name))

    return experiments


def main():
    """메인 자동 실험 실행"""
    device = config.get_device()

    print(f"\n{'='*60}")
    print("AUTOMATED EXPERIMENT")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # 타임스탬프 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 데이터 로드 (opt1, opt2 모두 로드)
    print("\n[1/3] Loading Data...")

    def load_split(records, split_name, extraction_style):
        return load_or_extract_data(
            record_list=records,
            base_path=config.DATA_PATH,
            valid_leads=config.VALID_LEADS,
            out_len=config.SIGNAL_LENGTH,
            split_name=split_name,
            extraction_style=extraction_style
        )

    data_loaders = {}

    # opt2 (DAEAC style) - baseline용
    print("  Loading opt2 (DAEAC) data...")
    train_data_opt2 = load_split(config.DS1_TRAIN, "Train", "daeac")
    valid_data_opt2 = load_split(config.DS1_VALID, "Valid", "daeac")
    test_data_opt2 = load_split(config.DS2_TEST, "Test", "daeac")

    DatasetClass_opt2 = DATASET_REGISTRY.get("daeac")
    train_ds_opt2 = DatasetClass_opt2(*train_data_opt2)
    valid_ds_opt2 = DatasetClass_opt2(*valid_data_opt2)
    test_ds_opt2 = DatasetClass_opt2(*test_data_opt2)

    data_loaders["opt2"] = (
        DataLoader(train_ds_opt2, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(valid_ds_opt2, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(test_ds_opt2, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
    )

    # opt1 (Standard) - fusion 실험용
    print("  Loading opt1 (Standard) data...")
    train_data_opt1 = load_split(config.DS1_TRAIN, "Train", "default")
    valid_data_opt1 = load_split(config.DS1_VALID, "Valid", "default")
    test_data_opt1 = load_split(config.DS2_TEST, "Test", "default")

    DatasetClass_opt1 = DATASET_REGISTRY.get("ecg_standard")
    train_ds_opt1 = DatasetClass_opt1(*train_data_opt1)
    valid_ds_opt1 = DatasetClass_opt1(*valid_data_opt1)
    test_ds_opt1 = DatasetClass_opt1(*test_data_opt1)

    data_loaders["opt1"] = (
        DataLoader(train_ds_opt1, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(valid_ds_opt1, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(test_ds_opt1, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
    )

    # opt3 (Early fusion + Late fusion) - opt2 데이터 + opt1의 7차원 RR features
    print("  Loading opt3 (Early+Late fusion) data...")
    opt3_start = time.time()
    
    def match_rr_features_7d(opt2_data, opt1_data):
        """opt2 샘플에 대해 opt1에서 같은 patient_id, sample_id를 가진 7차원 RR features를 찾아 매칭"""
        opt2_pids = opt2_data[3]  # patient_ids
        opt2_sids = opt2_data[4]  # sample_ids
        opt1_pids = opt1_data[3]
        opt1_sids = opt1_data[4]
        opt1_rr_7d = opt1_data[2]  # 7차원 RR features
        
        # opt1에서 (patient_id, sample_id) -> index 매핑 생성
        opt1_key_to_idx = {}
        for idx, (pid, sid) in enumerate(zip(opt1_pids, opt1_sids)):
            key = (int(pid), int(sid) if isinstance(sid, (int, np.integer)) else sid)
            opt1_key_to_idx[key] = idx
        
        # opt2의 각 샘플에 대해 opt1에서 매칭된 7차원 RR features 찾기
        matched_rr_7d = []
        matched_indices = []
        
        for idx, (pid, sid) in enumerate(zip(opt2_pids, opt2_sids)):
            key = (int(pid), int(sid) if isinstance(sid, (int, np.integer)) else sid)
            if key in opt1_key_to_idx:
                opt1_idx = opt1_key_to_idx[key]
                matched_rr_7d.append(opt1_rr_7d[opt1_idx])
                matched_indices.append(idx)
            else:
                # 매칭 실패 - 이 샘플은 제외
                pass
        
        # 매칭된 샘플만 필터링
        matched_data = opt2_data[0][matched_indices]
        matched_labels = opt2_data[1][matched_indices]
        matched_rr_2d = opt2_data[2][matched_indices]
        matched_pids = opt2_data[3][matched_indices]
        matched_sids = opt2_data[4][matched_indices]
        
        matched_rr_7d = np.array(matched_rr_7d)
        
        print(f"    Matched {len(matched_indices)}/{len(opt2_data[0])} samples")
        
        return (matched_data, matched_labels, matched_rr_2d, matched_rr_7d, matched_pids, matched_sids)
    
    # Train 데이터 매칭
    train_opt3 = match_rr_features_7d(train_data_opt2, train_data_opt1)
    valid_opt3 = match_rr_features_7d(valid_data_opt2, valid_data_opt1)
    test_opt3 = match_rr_features_7d(test_data_opt2, test_data_opt1)
    
    DatasetClass_opt3 = DATASET_REGISTRY.get("opt3")
    train_ds_opt3 = DatasetClass_opt3(*train_opt3)
    valid_ds_opt3 = DatasetClass_opt3(*valid_opt3)
    test_ds_opt3 = DatasetClass_opt3(*test_opt3)
    
    data_loaders["opt3"] = (
        DataLoader(train_ds_opt3, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(valid_ds_opt3, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(test_ds_opt3, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
    )
    
    opt3_time = time.time() - opt3_start
    print(f"  [opt3 dataset creation: {opt3_time:.2f}s]")

    print(f"  Train: {len(train_ds_opt2)} samples (opt2), {len(train_ds_opt1)} samples (opt1), {len(train_ds_opt3)} samples (opt3)")
    print(f"  Valid: {len(valid_ds_opt2)} samples (opt2), {len(valid_ds_opt1)} samples (opt1), {len(valid_ds_opt3)} samples (opt3)")
    print(f"  Test:  {len(test_ds_opt2)} samples (opt2), {len(test_ds_opt1)} samples (opt1), {len(test_ds_opt3)} samples (opt3)")

    # 실험 목록 생성
    experiments = generate_experiments()
    print(f"\n[2/3] Running {len(experiments)} experiments...")

    # 엑셀 writer 초기화 (실험 시작 전에 미리 복사)
    template_path = "./model_fusion.xlsx"
    excel_writer = None
    cumulative_writer = None

    if os.path.exists(template_path):
        excel_path = os.path.join(OUTPUT_DIR, "autoexp_results.xlsx")
        excel_writer = ExcelResultWriter(template_path, excel_path, classes=config.CLASSES)

        cumulative_path = "./cumulative_results.xlsx"
        cumulative_writer = CumulativeExcelWriter(template_path, cumulative_path, classes=config.CLASSES)
        print(f"[CumulativeExcel] Current record count: {cumulative_writer.get_record_count()}")

    # 실험 실행
    all_results = []
    total_start = time.time()

    for i, (exp_config, exp_name) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp_name}")

        try:
            result = run_single_experiment(exp_config, exp_name, data_loaders, device)
            all_results.append(result)

            # 실험 완료 즉시 엑셀에 기록
            if result['status'] == 'success' and 'full_metrics' in result:
                for metric_name in ["macro_auprc", "macro_auroc", "macro_recall"]:
                    if metric_name in result['full_metrics']:
                        metrics = result['full_metrics'][metric_name]
                        short_name = metric_name.replace("macro_", "")

                        if excel_writer:
                            excel_writer.write_metrics(exp_name, metrics, short_name)
                            if 'confusion_matrix' in metrics:
                                excel_writer.write_confusion_matrix(exp_name, metrics['confusion_matrix'], short_name)

                        if cumulative_writer:
                            cumulative_writer.append_result(exp_name, metrics, exp_config, short_name)
                            if 'confusion_matrix' in metrics:
                                cumulative_writer.append_confusion_matrix(exp_name, metrics['confusion_matrix'], short_name)

                print(f"  [Excel] Results saved for {exp_name}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                'exp_name': exp_name,
                'config': exp_config,
                'status': 'error',
                'error': str(e)
            })

    # 결과 요약
    total_time = (time.time() - total_start) / 60

    print(f"\n{'='*60}")
    print("[3/3] EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} min")
    print(f"Results saved to: {OUTPUT_DIR}")

    print(f"\n{'Experiment':<20} {'Acc':>8} {'F1':>8} {'AUPRC':>8} {'AUROC':>8}")
    print("-" * 56)

    for result in all_results:
        if result['status'] == 'success':
            r = result['test_macro_auprc']  # Best AUPRC 모델 결과
            print(f"{result['exp_name']:<20} {r['acc']:>8.4f} {r['macro_f1']:>8.4f} "
                  f"{r['macro_auprc']:>8.4f} {r['macro_auroc']:>8.4f}")
        else:
            print(f"{result['exp_name']:<20} FAILED")

    # 최종 요약 출력
    if excel_writer:
        print(f"\nExcel results saved to: {os.path.join(OUTPUT_DIR, 'autoexp_results.xlsx')}")
    if cumulative_writer:
        print(f"\nCumulative results saved to: ./cumulative_results.xlsx")
        print(f"[CumulativeExcel] Final record count: {cumulative_writer.get_record_count()}")


if __name__ == '__main__':
    main()
