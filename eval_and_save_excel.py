"""
저장된 모델을 평가하여 Excel에 저장하는 스크립트
"""

import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.registry import MODEL_REGISTRY, LOSS_REGISTRY, DATASET_REGISTRY
from src import models, losses, datasets
from src.trainer import Trainer
from src.utils import ExcelResultWriter
from utils import set_seed, load_or_extract_data

# 실험 결과 디렉토리
RESULTS_DIR = "/home/work/Ryuha/autoexp_results/20260124_144222"

# 엑셀 저장 경로
EXCEL_PATH = "/home/work/Ryuha/ECG_CrossAttention-stored/model_fusion_results_full.xlsx"
TEMPLATE_PATH = "/home/work/Ryuha/ECG_CrossAttention-stored/model_fusion.xlsx"

# 실험 설정 (main_autoexp.py와 동일)
EXPERIMENTS = [
    # (exp_name, exp_config)
    ("baseline_opt2", {"fusion_type": None, "lead": 3, "rr_dim": 2}),
    ("fusion_concat", {"fusion_type": "concat", "lead": 1, "rr_dim": 7}),
    ("opt3_concat", {"fusion_type": "concat", "lead": 3, "rr_dim": 7, "use_opt3": True}),
    ("fusion_concat_proj", {"fusion_type": "concat_proj", "lead": 1, "rr_dim": 7}),
    ("opt3_concat_proj", {"fusion_type": "concat_proj", "lead": 3, "rr_dim": 7, "use_opt3": True}),
    ("mhca_h1", {"fusion_type": "mhca", "fusion_num_heads": 1, "lead": 1, "rr_dim": 7}),
    ("mhca_h2", {"fusion_type": "mhca", "fusion_num_heads": 2, "lead": 1, "rr_dim": 7}),
    ("mhca_h4", {"fusion_type": "mhca", "fusion_num_heads": 4, "lead": 1, "rr_dim": 7}),
    ("mhca_h8", {"fusion_type": "mhca", "fusion_num_heads": 8, "lead": 1, "rr_dim": 7}),
    ("opt3_mhca_h1", {"fusion_type": "mhca", "fusion_num_heads": 1, "lead": 3, "rr_dim": 7, "use_opt3": True}),
    ("opt3_mhca_h2", {"fusion_type": "mhca", "fusion_num_heads": 2, "lead": 3, "rr_dim": 7, "use_opt3": True}),
    ("opt3_mhca_h4", {"fusion_type": "mhca", "fusion_num_heads": 4, "lead": 3, "rr_dim": 7, "use_opt3": True}),
    ("opt3_mhca_h8", {"fusion_type": "mhca", "fusion_num_heads": 8, "lead": 3, "rr_dim": 7, "use_opt3": True}),
]


def create_model(exp_config):
    """실험 설정에 맞는 모델 생성"""
    model_config = copy.deepcopy(config.MODEL_CONFIG)
    model_config.update(exp_config)

    ModelClass = MODEL_REGISTRY.get(config.MODEL_NAME)
    model = ModelClass(**model_config)
    return model


def load_split(records, split_name, extraction_style):
    """데이터 split 로드"""
    return load_or_extract_data(
        record_list=records,
        base_path=config.DATA_PATH,
        valid_leads=config.VALID_LEADS,
        out_len=config.SIGNAL_LENGTH,
        split_name=split_name,
        extraction_style=extraction_style
    )


def match_rr_features_7d(opt2_data, opt1_data):
    """opt2 샘플에 대해 opt1에서 같은 patient_id, sample_id를 가진 7차원 RR features를 찾아 매칭"""
    opt2_pids = opt2_data[3]
    opt2_sids = opt2_data[4]
    opt1_pids = opt1_data[3]
    opt1_sids = opt1_data[4]
    opt1_rr_7d = opt1_data[2]

    opt1_key_to_idx = {}
    for idx, (pid, sid) in enumerate(zip(opt1_pids, opt1_sids)):
        key = (int(pid), int(sid) if isinstance(sid, (int, np.integer)) else sid)
        opt1_key_to_idx[key] = idx

    matched_rr_7d = []
    matched_indices = []

    for idx, (pid, sid) in enumerate(zip(opt2_pids, opt2_sids)):
        key = (int(pid), int(sid) if isinstance(sid, (int, np.integer)) else sid)
        if key in opt1_key_to_idx:
            opt1_idx = opt1_key_to_idx[key]
            matched_rr_7d.append(opt1_rr_7d[opt1_idx])
            matched_indices.append(idx)

    matched_data = opt2_data[0][matched_indices]
    matched_labels = opt2_data[1][matched_indices]
    matched_rr_2d = opt2_data[2][matched_indices]
    matched_pids = opt2_data[3][matched_indices]
    matched_sids = opt2_data[4][matched_indices]
    matched_rr_7d = np.array(matched_rr_7d)

    print(f"    Matched {len(matched_indices)}/{len(opt2_data[0])} samples")

    return (matched_data, matched_labels, matched_rr_2d, matched_rr_7d, matched_pids, matched_sids)


def load_all_data():
    """모든 데이터 로드 (opt1, opt2, opt3)"""
    print("\n[1/3] Loading Data...")

    # opt2 (DAEAC style)
    print("  Loading opt2 (DAEAC) data...")
    train_data_opt2 = load_split(config.DS1_TRAIN, "Train", "daeac")
    valid_data_opt2 = load_split(config.DS1_VALID, "Valid", "daeac")
    test_data_opt2 = load_split(config.DS2_TEST, "Test", "daeac")

    # opt1 (Standard)
    print("  Loading opt1 (Standard) data...")
    train_data_opt1 = load_split(config.DS1_TRAIN, "Train", "default")
    valid_data_opt1 = load_split(config.DS1_VALID, "Valid", "default")
    test_data_opt1 = load_split(config.DS2_TEST, "Test", "default")

    return {
        "opt2": (train_data_opt2, valid_data_opt2, test_data_opt2),
        "opt1": (train_data_opt1, valid_data_opt1, test_data_opt1),
    }


def create_dataloaders(data_dict):
    """데이터로더 생성"""
    train_data_opt2, valid_data_opt2, test_data_opt2 = data_dict["opt2"]
    train_data_opt1, valid_data_opt1, test_data_opt1 = data_dict["opt1"]

    # opt2
    DatasetClass_opt2 = DATASET_REGISTRY.get("daeac")
    train_ds_opt2 = DatasetClass_opt2(*train_data_opt2)
    valid_ds_opt2 = DatasetClass_opt2(*valid_data_opt2)
    test_ds_opt2 = DatasetClass_opt2(*test_data_opt2)

    loaders_opt2 = (
        DataLoader(train_ds_opt2, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(valid_ds_opt2, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(test_ds_opt2, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
    )

    # opt1
    DatasetClass_opt1 = DATASET_REGISTRY.get("ecg_standard")
    train_ds_opt1 = DatasetClass_opt1(*train_data_opt1)
    valid_ds_opt1 = DatasetClass_opt1(*valid_data_opt1)
    test_ds_opt1 = DatasetClass_opt1(*test_data_opt1)

    loaders_opt1 = (
        DataLoader(train_ds_opt1, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(valid_ds_opt1, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(test_ds_opt1, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
    )

    # opt3 (Early fusion + Late fusion)
    print("  Creating opt3 (Early+Late fusion) datasets...")
    train_opt3 = match_rr_features_7d(train_data_opt2, train_data_opt1)
    valid_opt3 = match_rr_features_7d(valid_data_opt2, valid_data_opt1)
    test_opt3 = match_rr_features_7d(test_data_opt2, test_data_opt1)

    DatasetClass_opt3 = DATASET_REGISTRY.get("opt3")
    train_ds_opt3 = DatasetClass_opt3(*train_opt3)
    valid_ds_opt3 = DatasetClass_opt3(*valid_opt3)
    test_ds_opt3 = DatasetClass_opt3(*test_opt3)

    loaders_opt3 = (
        DataLoader(train_ds_opt3, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(valid_ds_opt3, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
        DataLoader(test_ds_opt3, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True),
    )

    print(f"  Test: {len(test_ds_opt2)} samples (opt2), {len(test_ds_opt1)} samples (opt1), {len(test_ds_opt3)} samples (opt3)")

    return {
        "opt1": loaders_opt1,
        "opt2": loaders_opt2,
        "opt3": loaders_opt3,
    }


def evaluate_experiment(exp_name, exp_config, data_loaders, device):
    """단일 실험 평가"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_name}")
    print(f"Config: {exp_config}")
    print(f"{'='*60}")

    # 모델 생성
    model = create_model(exp_config)
    model.to(device)

    # 데이터로더 선택
    fusion_type = exp_config.get('fusion_type')
    is_opt3 = exp_config.get('use_opt3', False)

    if is_opt3:
        loaders = data_loaders["opt3"]
    elif fusion_type is not None:
        loaders = data_loaders["opt1"]
    else:
        loaders = data_loaders["opt2"]

    train_loader, valid_loader, test_loader = loaders

    # Loss
    labels = train_loader.dataset.labels
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    counts = np.bincount(labels.flatten().astype(np.int64), minlength=config.NUM_CLASSES)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * config.NUM_CLASSES
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    LossClass = LOSS_REGISTRY.get(config.LOSS_NAME)
    criterion = LossClass(weight=class_weights)

    # Trainer (평가용)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=torch.optim.Adam(model.parameters()),
        device=device,
        exp_dir=os.path.join(RESULTS_DIR, exp_name),
        class_names=config.CLASSES,
    )

    # 각 best model 평가
    results = {}
    for metric in ["macro_auprc", "macro_auroc", "macro_recall"]:
        model_path = os.path.join(RESULTS_DIR, exp_name, "best_models", f"best_{metric}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded: {metric} (epoch {checkpoint.get('epoch', '?')})")

            test_metrics = trainer.evaluate(test_loader)
            results[metric] = test_metrics

            print(f"    Acc: {test_metrics['acc']:.4f}, "
                  f"F1: {test_metrics['macro_f1']:.4f}, "
                  f"AUPRC: {test_metrics['macro_auprc']:.4f}, "
                  f"AUROC: {test_metrics['macro_auroc']:.4f}")
        else:
            print(f"  Model not found: {model_path}")

    return results


def main():
    print("=" * 60)
    print("EVALUATING SAVED MODELS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(config.SEED)

    # 데이터 로드
    data_dict = load_all_data()
    data_loaders = create_dataloaders(data_dict)

    # Excel writer
    excel_writer = ExcelResultWriter(TEMPLATE_PATH, EXCEL_PATH, classes=config.CLASSES)

    # 각 실험 평가
    all_results = {}
    for exp_name, exp_config in EXPERIMENTS:
        try:
            results = evaluate_experiment(exp_name, exp_config, data_loaders, device)
            all_results[exp_name] = results

            # best_macro_auprc 기준으로 Excel 저장
            if "macro_auprc" in results:
                metrics = results["macro_auprc"]
                short_name = exp_name
                excel_writer.write_metrics(exp_name, metrics, short_name)
                if 'confusion_matrix' in metrics:
                    excel_writer.write_confusion_matrix(exp_name, metrics['confusion_matrix'], short_name)
                print(f"  [Excel] Saved: {exp_name}")
        except Exception as e:
            print(f"  [ERROR] {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results saved to: {EXCEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
