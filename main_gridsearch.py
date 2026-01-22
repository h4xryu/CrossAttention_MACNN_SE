"""
Grid Search Script

하이퍼파라미터 그리드 서치를 수행합니다.
EXPERIMENT_GRID의 모든 조합을 실험합니다.

사용법:
    python main_gridsearch.py
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
from utils import set_seed, load_or_extract_data


# =============================================================================
# 그리드 서치 설정 - 여기만 수정하면 됩니다
# =============================================================================

EXPERIMENT_GRID = {
    # 모델 구조
    "reduction": [8, 16],
    "dilations": [(1, 6, 12, 18), (1, 3, 6, 12)],

    # 학습률
    "lr": [0.005, 0.001],
}

# 실험별 epochs
EXP_EPOCHS = 50

# 결과 저장 경로
OUTPUT_DIR = "./gridsearch_results/"


# =============================================================================
# 실험 실행 함수
# =============================================================================

def run_single_experiment(exp_config: dict, exp_name: str, data_loaders: tuple, device):
    """단일 실험 수행"""
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

    # exp_config에서 모델 관련 설정만 추출
    model_keys = ["reduction", "dilations", "dropout", "fusion_type",
                  "fusion_emb", "fusion_expansion", "fusion_num_heads"]
    for key in model_keys:
        if key in exp_config:
            model_config[key] = exp_config[key]

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

    # Optimizer (lr을 exp_config에서 가져옴)
    opt_config = copy.deepcopy(config.OPTIMIZER_CONFIG)
    if "lr" in exp_config:
        opt_config["lr"] = exp_config["lr"]

    optimizer = build_optimizer(model.parameters(), config.OPTIMIZER_NAME, **opt_config)
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

    # 테스트 (Best AUPRC 모델)
    trainer.load_best_model("macro_auprc")
    test_metrics = trainer.evaluate(test_loader)

    return {
        'exp_name': exp_name,
        'config': exp_config,
        'test_acc': test_metrics['acc'],
        'test_f1': test_metrics['macro_f1'],
        'test_auprc': test_metrics['macro_auprc'],
        'test_auroc': test_metrics['macro_auroc'],
        'status': 'success'
    }


def generate_grid_experiments():
    """그리드 서치 조합 생성"""
    keys = list(EXPERIMENT_GRID.keys())
    values = list(EXPERIMENT_GRID.values())

    experiments = []
    for combination in product(*values):
        exp_config = dict(zip(keys, combination))

        # 실험 이름 생성
        name_parts = []
        for k, v in exp_config.items():
            if isinstance(v, tuple):
                name_parts.append(f"{k}={v[0]}-{v[-1]}")
            else:
                name_parts.append(f"{k}={v}")
        exp_name = "_".join(name_parts)

        experiments.append((exp_config, exp_name))

    return experiments


def main():
    """메인 그리드 서치 실행"""
    device = config.get_device()

    print(f"\n{'='*60}")
    print("GRID SEARCH")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Grid: {EXPERIMENT_GRID}")

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

    # Dataset & DataLoader
    DatasetClass = DATASET_REGISTRY.get(config.DATASET_NAME)
    train_dataset = DatasetClass(*train_data)
    valid_dataset = DatasetClass(*valid_data)
    test_dataset = DatasetClass(*test_data)

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
    experiments = generate_grid_experiments()
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
            import traceback
            traceback.print_exc()
            all_results.append({
                'exp_name': exp_name,
                'config': exp_config,
                'status': 'error',
                'error': str(e)
            })

    # 결과 요약
    total_time = (time.time() - total_start) / 60

    print(f"\n{'='*60}")
    print("[3/3] GRID SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} min")

    # 성공한 실험만 정렬
    successful = [r for r in all_results if r['status'] == 'success']
    successful.sort(key=lambda x: x['test_auprc'], reverse=True)

    print(f"\n{'Rank':<5} {'Experiment':<40} {'Acc':>8} {'F1':>8} {'AUPRC':>8}")
    print("-" * 75)

    for i, result in enumerate(successful[:10], 1):  # Top 10
        print(f"{i:<5} {result['exp_name']:<40} {result['test_acc']:>8.4f} "
              f"{result['test_f1']:>8.4f} {result['test_auprc']:>8.4f}")

    # Best 설정 출력
    if successful:
        best = successful[0]
        print(f"\n{'='*60}")
        print("BEST CONFIGURATION")
        print(f"{'='*60}")
        for k, v in best['config'].items():
            print(f"  {k}: {v}")
        print(f"\n  Test Acc: {best['test_acc']:.4f}")
        print(f"  Test F1:  {best['test_f1']:.4f}")
        print(f"  Test AUPRC: {best['test_auprc']:.4f}")

    # 결과 저장
    import json
    results_path = os.path.join(OUTPUT_DIR, "gridsearch_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
