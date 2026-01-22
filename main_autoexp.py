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
from src.utils import ExcelResultWriter
from utils import set_seed, load_or_extract_data


# =============================================================================
# 실험 설정 - 여기만 수정하면 됩니다
# =============================================================================

# 실험할 파라미터 그리드
EXPERIMENT_GRID = {
    # Fusion type 실험 (opt1, lead=1일 때만 의미있음)
    "fusion_type": [None, "concat", "concat_proj", "mhca"],

    # MHCA용 num_heads (fusion_type="mhca"일 때만 사용)
    "fusion_num_heads": [1, 2, 4],
}

# 실험별 epochs (빠른 실험용)
EXP_EPOCHS = 50

# 결과 저장 경로
OUTPUT_DIR = "./autoexp_results/"


# =============================================================================
# 실험 실행 함수
# =============================================================================

def run_single_experiment(exp_config: dict, exp_name: str, data_loaders: tuple, device):
    """
    단일 실험 수행

    Args:
        exp_config: 실험 설정 dict (MODEL_CONFIG에 병합됨)
        exp_name: 실험 이름
        data_loaders: (train_loader, valid_loader, test_loader)
        device: torch device

    Returns:
        결과 dict
    """
    train_loader, valid_loader, test_loader = data_loaders

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
        if fusion_type == "mhca":
            # MHCA는 num_heads별로 실험
            for num_heads in EXPERIMENT_GRID["fusion_num_heads"]:
                exp_config = {
                    "fusion_type": fusion_type,
                    "fusion_num_heads": num_heads,
                }
                exp_name = f"mhca_h{num_heads}"
                experiments.append((exp_config, exp_name))
        else:
            exp_config = {"fusion_type": fusion_type}
            exp_name = f"fusion_{fusion_type}" if fusion_type else "baseline"
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

    # 데이터 로드 (한 번만)
    print("\n[1/3] Loading Data...")

    def load_split(records, split_name):
        return load_or_extract_data(
            record_list=records,
            base_path=config.DATA_PATH,
            valid_leads=config.VALID_LEADS,
            out_len=config.SIGNAL_LENGTH,
            split_name=split_name,
            extraction_style=config.EXTRACTION_STYLE
        )

    train_data = load_split(config.DS1_TRAIN, "Train")
    valid_data = load_split(config.DS1_VALID, "Valid")
    test_data = load_split(config.DS2_TEST, "Test")

    # Dataset 생성
    DatasetClass = DATASET_REGISTRY.get(config.DATASET_NAME)
    train_dataset = DatasetClass(*train_data)
    valid_dataset = DatasetClass(*valid_data)
    test_dataset = DatasetClass(*test_data)

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    data_loaders = (train_loader, valid_loader, test_loader)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Valid: {len(valid_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # 실험 목록 생성
    experiments = generate_experiments()
    print(f"\n[2/3] Running {len(experiments)} experiments...")

    # 실험 실행
    all_results = []
    total_start = time.time()

    for i, (exp_config, exp_name) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp_name}")

        try:
            result = run_single_experiment(exp_config, exp_name, data_loaders, device)
            all_results.append(result)
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

    # JSON 결과 저장
    import json
    results_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # 엑셀 결과 저장 (템플릿이 있는 경우)
    template_path = "./model_fusion.xlsx"
    if os.path.exists(template_path):
        excel_path = os.path.join(OUTPUT_DIR, "autoexp_results.xlsx")
        excel_writer = ExcelResultWriter(template_path, excel_path, classes=config.CLASSES)

        for result in all_results:
            if result['status'] == 'success' and 'full_metrics' in result:
                exp_name = result['exp_name']
                # 모든 best model 결과 저장 (auprc, auroc, recall)
                for metric_name in ["macro_auprc", "macro_auroc", "macro_recall"]:
                    if metric_name in result['full_metrics']:
                        metrics = result['full_metrics'][metric_name]
                        short_name = metric_name.replace("macro_", "")  # auprc, auroc, recall
                        excel_writer.write_metrics(exp_name, metrics, short_name)
                        if 'confusion_matrix' in metrics:
                            excel_writer.write_confusion_matrix(exp_name, metrics['confusion_matrix'], short_name)

        print(f"Excel results saved to: {excel_path}")

    print(f"\nJSON results saved to: {results_path}")


if __name__ == '__main__':
    main()
