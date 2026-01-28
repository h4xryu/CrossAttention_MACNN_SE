# main_autoexp_for_ResU.py - ResU 모델 자동 실험
# opt1 (1D) / opt3 (2D) 지원
# AUROC, AUPRC, Last 3가지 best model 저장 후 테스트

import os
import sys
import time
import copy
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

# ResU 모델 import
sys.path.append('./src/models')
from ResU import get_resu_model

# =============================================================================
# 설정
# =============================================================================

DATA_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_PATH = './auto_results_ResU/'
CACHE_DIR = './dataset_cache_ResU/'

BATCH_SIZE = 1024
EPOCHS = 75
LR = 0.0001
WEIGHT_DECAY = 1e-2
SEED = 1234
CLASSES = ['N', 'S', 'V', 'F']

# 모델 설정
MODEL_CONFIG = {
    'in_channels': 1,
    'out_ch': 128,
    'mid_ch': 32,
    'num_heads': 1,
    # opt1: 7차원 RR features (pre, post, local, global, ratios)
    # opt3: 2차원 RR features (pre_rr_ratio, near_pre_rr_ratio) - 3채널 입력에 포함
    'n_rr_opt1': 7,
    'n_rr_opt3': 2,
    'n_channels_opt3': 3,  # opt3: ECG + pre_rr_ratio + near_pre_rr_ratio
}

# ECG Parameters
OUT_LEN = 720

# 데이터 분할
DS1_TRAIN = [
    '101', '106', '108', '109', '112', '115', '116', '118', '119',
    '122', '201', '203', '209', '215', '223', '230'
]
DS1_VALID = ['114', '124', '205', '207', '220', '208']
DS2_TEST = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]

# 실험 정의
# (실험명, 모델타입, 입력모드)
EXPERIMENTS = [
    # opt1 (1D): baseline, naive_concat, cross_attention
    ('ResU_opt1_baseline', 'baseline', 'opt1'),
    ('ResU_opt1_naive', 'naive_concat', 'opt1'),
    ('ResU_opt1_cross', 'cross_attention', 'opt1'),

    # opt3 (2D): baseline, naive_concat, cross_attention
    ('ResU_opt3_baseline', 'baseline', 'opt3'),
    ('ResU_opt3_naive', 'naive_concat', 'opt3'),
    ('ResU_opt3_cross', 'cross_attention', 'opt3'),
]

# =============================================================================
# Seed 설정
# =============================================================================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# =============================================================================
# 데이터 로드 (wfdb 사용)
# =============================================================================
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt
from tqdm import tqdm
import hashlib
import json

LABEL_GROUP_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
}
LABEL_TO_ID = {'N': 0, 'S': 1, 'V': 2, 'F': 3}


def resample_to_len(ts: np.ndarray, out_len: int) -> np.ndarray:
    if len(ts) == out_len:
        return ts.astype(np.float32)
    if len(ts) < 2:
        return np.pad(ts.astype(np.float32), (0, max(0, out_len - len(ts))), mode='edge')[:out_len]
    x_old = np.linspace(0.0, 1.0, num=len(ts))
    x_new = np.linspace(0.0, 1.0, num=out_len)
    return interp1d(x_old, ts, kind='linear')(x_new).astype(np.float32)


def remove_baseline_bandpass(signal: np.ndarray, fs: int = 360,
                              lowcut: float = 0.1, highcut: float = 100.0,
                              order: int = 256) -> np.ndarray:
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    fir_coeff = firwin(order + 1, [low, high], pass_zero=False)
    return filtfilt(fir_coeff, 1.0, signal)


def compute_rr_features(ann_samples, center_idx, fs=360):
    """7차원 RR features 계산"""
    rr_intervals = np.diff(ann_samples) / fs * 1000  # ms 단위

    if len(rr_intervals) < 2:
        return np.zeros(7, dtype=np.float32)

    # 현재 beat 기준 전후 RR intervals
    beat_idx = center_idx

    # 전 RR (pre-RR)
    if beat_idx > 0:
        pre_rr = rr_intervals[beat_idx - 1]
    else:
        pre_rr = np.mean(rr_intervals)

    # 후 RR (post-RR)
    if beat_idx < len(rr_intervals):
        post_rr = rr_intervals[beat_idx]
    else:
        post_rr = np.mean(rr_intervals)

    # Local RR (±5 beats 평균)
    local_start = max(0, beat_idx - 5)
    local_end = min(len(rr_intervals), beat_idx + 5)
    local_rr = np.mean(rr_intervals[local_start:local_end]) if local_end > local_start else pre_rr

    # Global RR (전체 평균)
    global_rr = np.mean(rr_intervals)

    # RR ratios
    pre_ratio = pre_rr / (local_rr + 1e-8)
    post_ratio = post_rr / (local_rr + 1e-8)
    local_global_ratio = local_rr / (global_rr + 1e-8)

    return np.array([pre_rr, post_rr, local_rr, global_rr,
                     pre_ratio, post_ratio, local_global_ratio], dtype=np.float32)


def compute_rr_features_2d(ann_samples, center_idx, fs=360):
    """
    opt3용 2차원 RR features 계산 (DAEACDataset 스타일)
    Returns: [pre_rr_ratio, near_pre_rr_ratio]
    """
    rr_intervals = np.diff(ann_samples) / fs * 1000  # ms 단위

    if len(rr_intervals) < 2:
        return np.array([1.0, 1.0], dtype=np.float32)

    beat_idx = center_idx

    # pre-RR interval
    if beat_idx > 0:
        pre_rr = rr_intervals[beat_idx - 1]
    else:
        pre_rr = np.mean(rr_intervals)

    # near pre-RR (post-RR of previous beat = pre-RR of current beat's neighbor)
    if beat_idx > 1:
        near_pre_rr = rr_intervals[beat_idx - 2]
    elif beat_idx > 0:
        near_pre_rr = rr_intervals[beat_idx - 1]
    else:
        near_pre_rr = np.mean(rr_intervals)

    # Local RR average (±5 beats)
    local_start = max(0, beat_idx - 5)
    local_end = min(len(rr_intervals), beat_idx + 5)
    local_rr = np.mean(rr_intervals[local_start:local_end]) if local_end > local_start else pre_rr

    # Ratios
    pre_rr_ratio = pre_rr / (local_rr + 1e-8)
    near_pre_rr_ratio = near_pre_rr / (local_rr + 1e-8)

    return np.array([pre_rr_ratio, near_pre_rr_ratio], dtype=np.float32)


def extract_data_opt1(record_list, base_path, out_len, split_name):
    """
    opt1: 1D single channel ECG + 7-dim RR features
    Returns: (ecg_data, labels, rr_features, patient_ids, sample_ids)
    """
    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}

    all_ecg = []
    all_labels = []
    all_rr = []
    all_pids = []
    all_sids = []

    for rec in tqdm(record_list, desc=f"Extracting {split_name} (opt1)"):
        try:
            ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
            sig, meta = wfdb.rdsamp(os.path.join(base_path, rec))
        except Exception as e:
            print(f"Warning: {rec} failed - {e}")
            continue

        patient_id = patient_to_id[rec]
        fs = int(meta['fs'])
        sig_names = meta['sig_name']

        # MLII 선택
        if 'MLII' in sig_names:
            ch_idx = sig_names.index('MLII')
        else:
            ch_idx = 0

        ecg_raw = sig[:, ch_idx].astype(np.float32)
        ecg_filtered = remove_baseline_bandpass(ecg_raw, fs=fs)

        pre = int(round(360 * fs / 360.0))
        post = int(round(360 * fs / 360.0))

        for idx, (center, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            grp = LABEL_GROUP_MAP.get(symbol, None)
            if grp is None or grp not in CLASSES:
                continue

            start = center - pre
            end = center + post

            if start < 0 or end > len(ecg_filtered):
                continue

            seg = ecg_filtered[start:end]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            seg_resampled = resample_to_len(seg, out_len)

            rr_feat = compute_rr_features(ann.sample, idx, fs)

            all_ecg.append(seg_resampled[np.newaxis, :])  # (1, L)
            all_labels.append(LABEL_TO_ID[grp])
            all_rr.append(rr_feat)
            all_pids.append(patient_id)
            all_sids.append(idx)

    print(f"{split_name} (opt1): {len(all_ecg)} samples")

    return (
        np.array(all_ecg, dtype=np.float32),
        np.array(all_labels, dtype=np.int64),
        np.array(all_rr, dtype=np.float32),
        np.array(all_pids, dtype=np.int64),
        np.array(all_sids, dtype=np.int64)
    )


def extract_data_opt3(record_list, base_path, out_len, split_name):
    """
    opt3: DAEAC 스타일 3채널 입력 (ECG + pre_rr_ratio + near_pre_rr_ratio)
    Returns: (ecg_data, labels, rr_features, patient_ids, sample_ids)

    ecg_data shape: (N, L) - ECG 시그널만 (Dataset에서 3채널로 조합)
    rr_features shape: (N, 2) - [pre_rr_ratio, near_pre_rr_ratio]
    """
    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}

    all_ecg = []
    all_labels = []
    all_rr = []
    all_pids = []
    all_sids = []

    for rec in tqdm(record_list, desc=f"Extracting {split_name} (opt3)"):
        try:
            ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
            sig, meta = wfdb.rdsamp(os.path.join(base_path, rec))
        except Exception as e:
            print(f"Warning: {rec} failed - {e}")
            continue

        patient_id = patient_to_id[rec]
        fs = int(meta['fs'])
        sig_names = meta['sig_name']

        # MLII 채널 선택 (단일 채널)
        if 'MLII' in sig_names:
            ch_idx = sig_names.index('MLII')
        else:
            ch_idx = 0

        ecg_raw = sig[:, ch_idx].astype(np.float32)
        ecg_filtered = remove_baseline_bandpass(ecg_raw, fs=fs)

        pre = int(round(360 * fs / 360.0))
        post = int(round(360 * fs / 360.0))

        for idx, (center, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            grp = LABEL_GROUP_MAP.get(symbol, None)
            if grp is None or grp not in CLASSES:
                continue

            start = center - pre
            end = center + post

            if start < 0 or end > len(ecg_filtered):
                continue

            seg = ecg_filtered[start:end]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            seg_resampled = resample_to_len(seg, out_len)

            # 2차원 RR features (DAEAC 스타일)
            rr_feat = compute_rr_features_2d(ann.sample, idx, fs)

            all_ecg.append(seg_resampled)  # (L,) - Dataset에서 3채널로 조합
            all_labels.append(LABEL_TO_ID[grp])
            all_rr.append(rr_feat)  # (2,)
            all_pids.append(patient_id)
            all_sids.append(idx)

    print(f"{split_name} (opt3): {len(all_ecg)} samples")

    return (
        np.array(all_ecg, dtype=np.float32),  # (N, L)
        np.array(all_labels, dtype=np.int64),
        np.array(all_rr, dtype=np.float32),   # (N, 2)
        np.array(all_pids, dtype=np.int64),
        np.array(all_sids, dtype=np.int64)
    )


# =============================================================================
# Dataset
# =============================================================================
class ECGDataset_opt1(Dataset):
    """
    opt1용 Dataset: 1D ECG (1, L) + 7-dim RR features
    """
    def __init__(self, ecg_data, labels, rr_features, patient_ids, sample_ids):
        # ecg_data: (N, 1, L)
        self.ecg_data = torch.FloatTensor(ecg_data)
        self.labels = torch.LongTensor(labels)
        self.rr_features = torch.FloatTensor(rr_features)  # (N, 7)
        self.patient_ids = torch.LongTensor(patient_ids)
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.ecg_data[idx],       # (1, L)
            self.rr_features[idx],    # (7,)
            self.labels[idx],
            self.patient_ids[idx],
            idx
        )


class ECGDataset_opt3(Dataset):
    """
    opt3용 Dataset: DAEAC 스타일 3채널 입력
    ECG + pre_rr_ratio + near_pre_rr_ratio → (1, 3, L)
    """
    def __init__(self, ecg_data, labels, rr_features, patient_ids, sample_ids):
        # ecg_data: (N, L) - ECG 시그널
        # rr_features: (N, 2) - [pre_rr_ratio, near_pre_rr_ratio]
        self.ecg_data = ecg_data  # numpy array (N, L)
        self.labels = torch.LongTensor(labels)
        self.rr_features = rr_features  # numpy array (N, 2)
        self.patient_ids = torch.LongTensor(patient_ids)
        self.sample_ids = sample_ids
        self.seq_len = ecg_data.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ecg = self.ecg_data[idx]  # (L,)
        rr = self.rr_features[idx]  # (2,)

        # RR features를 ECG 길이로 확장
        pre_rr_ratio = np.full(self.seq_len, rr[0], dtype=np.float32)
        near_pre_rr_ratio = np.full(self.seq_len, rr[1], dtype=np.float32)

        # 3채널로 스택: (3, L) → (1, 3, L)
        x = np.stack([ecg, pre_rr_ratio, near_pre_rr_ratio], axis=0)
        x = x[np.newaxis, :, :]  # (1, 3, L)

        return (
            torch.from_numpy(x).float(),  # (1, 3, L)
            torch.from_numpy(rr).float(),  # (2,) - 별도 RR features (cross-attention용)
            self.labels[idx],
            self.patient_ids[idx],
            idx
        )


# =============================================================================
# 학습/평가 함수
# =============================================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        ecg, rr, labels, _, _ = batch
        ecg = ecg.to(device)
        rr = rr.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(ecg, rr)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ecg.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {'loss': total_loss / total, 'acc': correct / total}


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            ecg, rr, labels, _, _ = batch
            ecg = ecg.to(device)
            rr = rr.to(device)
            labels = labels.to(device)

            logits, _ = model(ecg, rr)
            loss = criterion(logits, labels)

            total_loss += loss.item() * ecg.size(0)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # AUROC/AUPRC 계산
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    n_classes = len(CLASSES)
    y_true_onehot = np.eye(n_classes)[y_true]

    try:
        per_class_auroc = []
        per_class_auprc = []
        for c in range(n_classes):
            if y_true_onehot[:, c].sum() > 0:
                auroc = roc_auc_score(y_true_onehot[:, c], y_probs[:, c])
                auprc = average_precision_score(y_true_onehot[:, c], y_probs[:, c])
            else:
                auroc = 0.0
                auprc = 0.0
            per_class_auroc.append(auroc)
            per_class_auprc.append(auprc)

        macro_auroc = np.mean(per_class_auroc)
        macro_auprc = np.mean(per_class_auprc)
    except:
        macro_auroc = 0.0
        macro_auprc = 0.0
        per_class_auroc = [0.0] * n_classes
        per_class_auprc = [0.0] * n_classes

    metrics = {
        'loss': total_loss / total,
        'acc': correct / total,
        'macro_auroc': macro_auroc,
        'macro_auprc': macro_auprc,
        'per_class_auroc': per_class_auroc,
        'per_class_auprc': per_class_auprc,
    }

    return metrics, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_test_metrics(y_true, y_pred, y_probs, classes):
    """테스트용 상세 메트릭 계산"""
    n_classes = len(classes)

    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0
    )

    y_true_onehot = np.eye(n_classes)[y_true]

    try:
        per_class_auroc = []
        per_class_auprc = []
        for c in range(n_classes):
            if y_true_onehot[:, c].sum() > 0:
                auroc = roc_auc_score(y_true_onehot[:, c], y_probs[:, c])
                auprc = average_precision_score(y_true_onehot[:, c], y_probs[:, c])
            else:
                auroc = 0.0
                auprc = 0.0
            per_class_auroc.append(auroc)
            per_class_auprc.append(auprc)

        macro_auroc = np.mean(per_class_auroc)
        macro_auprc = np.mean(per_class_auprc)
    except:
        per_class_auroc = [0.0] * n_classes
        per_class_auprc = [0.0] * n_classes
        macro_auroc = 0.0
        macro_auprc = 0.0

    return {
        'confusion_matrix': cm,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_auroc': np.array(per_class_auroc),
        'per_class_auprc': np.array(per_class_auprc),
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1),
        'macro_auroc': macro_auroc,
        'macro_auprc': macro_auprc,
        'accuracy': (y_true == y_pred).mean(),
        'support': support,
    }


# =============================================================================
# 실험 실행
# =============================================================================
def run_experiment(exp_name, model_type, input_mode, data_dict, device):
    """단일 실험 수행"""
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"  Model: {model_type}, Input: {input_mode}")
    print(f"{'='*80}")

    set_seed(SEED)

    # 실험 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_PATH, f'{exp_name}_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'best_weights'), exist_ok=True)

    # 데이터 로드 및 Dataset 생성 (input_mode에 따라 다른 Dataset 사용)
    train_data = data_dict[input_mode]['train']
    valid_data = data_dict[input_mode]['valid']
    test_data = data_dict[input_mode]['test']

    if input_mode == 'opt1':
        # opt1: 1D ECG (1, L) + 7-dim RR features
        train_dataset = ECGDataset_opt1(*train_data)
        valid_dataset = ECGDataset_opt1(*valid_data)
        test_dataset = ECGDataset_opt1(*test_data)
        n_rr = MODEL_CONFIG['n_rr_opt1']  # 7
    else:  # opt3
        # opt3: DAEAC 스타일 3채널 (1, 3, L) + 2-dim RR features
        train_dataset = ECGDataset_opt3(*train_data)
        valid_dataset = ECGDataset_opt3(*valid_data)
        test_dataset = ECGDataset_opt3(*test_data)
        n_rr = MODEL_CONFIG['n_rr_opt3']  # 2

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_dataset):,} | Valid: {len(valid_dataset):,} | Test: {len(test_dataset):,}")

    # 모델 생성 (input_mode에 따라 다른 n_rr 사용)
    model_config = {
        'in_channels': MODEL_CONFIG['in_channels'],
        'out_ch': MODEL_CONFIG['out_ch'],
        'mid_ch': MODEL_CONFIG['mid_ch'],
        'num_heads': MODEL_CONFIG['num_heads'],
        'n_rr': n_rr,
    }
    if input_mode == 'opt3':
        model_config['n_channels_opt3'] = MODEL_CONFIG['n_channels_opt3']  # 3

    model = get_resu_model(
        model_type=model_type,
        input_mode=input_mode,
        nOUT=len(CLASSES),
        **model_config
    ).to(device)

    # 학습 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Best 모델 추적
    initial_state = copy.deepcopy(model.state_dict())
    best = {
        'auroc': {'value': -float('inf'), 'epoch': 0, 'state_dict': initial_state, 'valid_metrics': None},
        'auprc': {'value': -float('inf'), 'epoch': 0, 'state_dict': initial_state, 'valid_metrics': None},
        'last': {'epoch': EPOCHS, 'state_dict': initial_state, 'valid_metrics': None}
    }

    # 학습 루프
    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        valid_metrics, _, _, _ = validate(model, valid_loader, device)

        # Best 체크
        if valid_metrics['macro_auroc'] > best['auroc']['value']:
            best['auroc'] = {
                'value': valid_metrics['macro_auroc'],
                'epoch': epoch,
                'state_dict': copy.deepcopy(model.state_dict()),
                'valid_metrics': copy.deepcopy(valid_metrics)
            }

        if valid_metrics['macro_auprc'] > best['auprc']['value']:
            best['auprc'] = {
                'value': valid_metrics['macro_auprc'],
                'epoch': epoch,
                'state_dict': copy.deepcopy(model.state_dict()),
                'valid_metrics': copy.deepcopy(valid_metrics)
            }

        if epoch == EPOCHS:
            best['last'] = {
                'epoch': epoch,
                'state_dict': copy.deepcopy(model.state_dict()),
                'valid_metrics': copy.deepcopy(valid_metrics)
            }

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} | "
                  f"Valid Acc: {valid_metrics['acc']:.4f} AUROC: {valid_metrics['macro_auroc']:.4f} AUPRC: {valid_metrics['macro_auprc']:.4f}")

    # 테스트
    print(f"\n--- Testing Best Models ---")
    results_dict = {}

    for model_key in ['auroc', 'auprc', 'last']:
        epoch = best[model_key]['epoch']
        print(f"\n  [{model_key.upper()}] Epoch {epoch}")

        model.load_state_dict(best[model_key]['state_dict'])
        _, t_preds, t_labels, t_probs = validate(model, test_loader, device)
        test_metrics = calculate_test_metrics(t_labels, t_preds, t_probs, CLASSES)

        results_dict[model_key] = {
            'best_epoch': epoch,
            'valid_metrics': best[model_key]['valid_metrics'],
            'test_metrics': test_metrics
        }

        print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"    Macro AUROC: {test_metrics['macro_auroc']:.4f}")
        print(f"    Macro AUPRC: {test_metrics['macro_auprc']:.4f}")
        print(f"    Per-class Recall: {dict(zip(CLASSES, test_metrics['per_class_recall'].round(4)))}")

    # Best weights 저장
    for model_key in ['auroc', 'auprc', 'last']:
        torch.save({
            'model_state_dict': best[model_key]['state_dict'],
            'epoch': best[model_key]['epoch'],
        }, os.path.join(exp_dir, 'best_weights', f'best_{model_key}.pth'))

    print(f"\n  Results saved to: {exp_dir}")

    return results_dict, exp_dir


# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("ResU Automated Experiments")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 80)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 데이터 로드
    print("\n[1/2] Loading Data...")
    data_dict = {}

    # opt1 데이터
    print("\n  Loading opt1 (1D) data...")
    train_opt1 = extract_data_opt1(DS1_TRAIN, DATA_PATH, OUT_LEN, "Train")
    valid_opt1 = extract_data_opt1(DS1_VALID, DATA_PATH, OUT_LEN, "Valid")
    test_opt1 = extract_data_opt1(DS2_TEST, DATA_PATH, OUT_LEN, "Test")
    data_dict['opt1'] = {'train': train_opt1, 'valid': valid_opt1, 'test': test_opt1}

    # opt3 데이터 (DAEAC 스타일: ECG + RR ratios as 3 channels)
    print("\n  Loading opt3 (DAEAC style 3-channel) data...")
    train_opt3 = extract_data_opt3(DS1_TRAIN, DATA_PATH, OUT_LEN, "Train")
    valid_opt3 = extract_data_opt3(DS1_VALID, DATA_PATH, OUT_LEN, "Valid")
    test_opt3 = extract_data_opt3(DS2_TEST, DATA_PATH, OUT_LEN, "Test")
    data_dict['opt3'] = {'train': train_opt3, 'valid': valid_opt3, 'test': test_opt3}

    # 실험 실행
    print(f"\n[2/2] Running {len(EXPERIMENTS)} experiments...")
    all_results = {}

    for idx, (exp_name, model_type, input_mode) in enumerate(EXPERIMENTS):
        print(f"\n[{idx+1}/{len(EXPERIMENTS)}] {exp_name}")

        try:
            results_dict, exp_dir = run_experiment(exp_name, model_type, input_mode, data_dict, device)
            all_results[exp_name] = results_dict
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # 결과 요약
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\n{'Experiment':<25} {'Acc':>8} {'AUROC':>8} {'AUPRC':>8}")
    print("-" * 53)

    for exp_name, results in all_results.items():
        if 'auroc' in results:
            m = results['auroc']['test_metrics']
            print(f"{exp_name:<25} {m['accuracy']:>8.4f} {m['macro_auroc']:>8.4f} {m['macro_auprc']:>8.4f}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
