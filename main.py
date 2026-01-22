"""
ECG Classification - Main Training Script

config.py의 설정을 기반으로 학습을 실행합니다.

사용법:
    python main.py

설정 변경:
    config.py 파일을 수정하세요.
"""

import os
import numpy as np
import torch

# Config (전역 설정)
import config

# Registry imports (모든 컴포넌트 자동 등록)
from src.registry import MODEL_REGISTRY, LOSS_REGISTRY, DATASET_REGISTRY
from src import models      # 모델 등록
from src import losses      # Loss 등록
from src import datasets    # Dataset 등록
from src.optimizers import build_optimizer
from src.schedulers import build_scheduler
from src.trainer import Trainer

# Utils
from utils import set_seed, load_or_extract_data


def main():
    # =========================================================================
    # 1. Initialization
    # =========================================================================
    set_seed(config.SEED)
    device = config.get_device()
    exp_dir = config.create_experiment_dir()

    config.print_config()
    print(f"Device: {device}")
    print(f"Output: {exp_dir}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("\n[1/4] Loading Data...")

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

    # Create datasets
    DatasetClass = DATASET_REGISTRY.get(config.DATASET_NAME)
    train_dataset = DatasetClass(*train_data)
    valid_dataset = DatasetClass(*valid_data)
    test_dataset = DatasetClass(*test_data)

    # Create dataloaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Valid: {len(valid_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # =========================================================================
    # 3. Model
    # =========================================================================
    print("\n[2/4] Building Model...")

    ModelClass = MODEL_REGISTRY.get(config.MODEL_NAME)
    model = ModelClass(**config.MODEL_CONFIG)

    total_params, trainable_params = model.count_parameters()
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")

    # =========================================================================
    # 4. Loss Function
    # =========================================================================
    print("\n[3/4] Setting up Loss & Optimizer...")

    # Compute class weights if needed
    class_weights = None
    if config.LOSS_CONFIG.get('use_class_weights', False):
        labels = train_dataset.labels
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        else:
            labels = np.array(labels)

        counts = np.bincount(labels.flatten().astype(np.int64), minlength=config.NUM_CLASSES)
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * config.NUM_CLASSES
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"  Class counts: {counts}")
        print(f"  Class weights: {weights.round(4)}")

    # Create loss
    LossClass = LOSS_REGISTRY.get(config.LOSS_NAME)
    criterion = LossClass(
        weight=class_weights,
        label_smoothing=config.LOSS_CONFIG.get('label_smoothing', 0.0)
    )

    # =========================================================================
    # 5. Optimizer & Scheduler
    # =========================================================================
    optimizer = build_optimizer(
        model.parameters(),
        name=config.OPTIMIZER_NAME,
        **config.OPTIMIZER_CONFIG
    )

    scheduler = build_scheduler(
        optimizer,
        name=config.SCHEDULER_NAME,
        **config.SCHEDULER_CONFIG
    )

    print(f"  Loss: {config.LOSS_NAME}")
    print(f"  Optimizer: {config.OPTIMIZER_NAME} (lr={config.OPTIMIZER_CONFIG['lr']})")
    print(f"  Scheduler: {config.SCHEDULER_NAME}")

    # =========================================================================
    # 6. Training
    # =========================================================================
    print("\n[4/4] Training...")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        exp_dir=exp_dir,
        gradient_clip=config.GRADIENT_CLIP_VAL,
        save_every=config.SAVE_EVERY,
        class_names=config.CLASSES,
    )

    trainer.fit(
        epochs=config.EPOCHS,
        early_stopping=config.EARLY_STOPPING,
        patience=config.PATIENCE,
        monitor=config.MONITOR_METRIC
    )

    # =========================================================================
    # 7. Test Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    for metric in config.SAVE_BEST_METRICS:
        print(f"\n--- Best {metric.upper()} ---")
        trainer.load_best_model(metric)
        test_metrics = trainer.evaluate(test_loader)
        trainer.print_metrics(test_metrics, f"Test Results (best {metric})")

    # =========================================================================
    # 8. Save Final Results
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
