import os
import random
import hashlib
import json
import numpy as np
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from config import CLASSES, LABEL_TO_ID, LABEL_GROUP_MAP, RR_FEATURE_OPTION
from joblib import Parallel, delayed
import multiprocessing as mp
from typing import Tuple

# =============================================================================
# Dataset Cache Directory
# =============================================================================
CACHE_DIR = './dataset'

# =============================================================================
# Seed Setting
# =============================================================================

def set_seed(seed: int, fully_deterministic: bool = True) -> None:
    """
    랜덤 시드 설정

    Args:
        seed: 랜덤 시드
        fully_deterministic: True면 완전 결정론적 (느림), False면 기본 설정
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 기본 cuDNN 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Signal Processing
# =============================================================================

def resample_to_len(ts: np.ndarray, out_len: int) -> np.ndarray:
    """신호를 지정된 길이로 리샘플링"""
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
    """밴드패스 필터를 사용한 베이스라인 제거"""
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    fir_coeff = firwin(order + 1, [low, high], pass_zero=False)
    return filtfilt(fir_coeff, 1.0, signal)


# =============================================================================
# RR Feature Extraction Functions
# =============================================================================

def get_rr_feature_function(option: str):
    """
    RR feature 옵션에 따라 적절한 함수 반환

    Args:
        option: "opt1", "opt2", ... (확장 가능)

    Returns:
        feature extraction function

    옵션 설명:
        opt1: 기존 실험용 (7 features) - Cross-Attention 연구
        opt2: DAEAC 논문 재현용 (2 features)
        opt3+: 향후 새로운 연구용 (확장)
    """
    functions = {
        "opt1": compute_rr_features_opt1,  # 7 features (기존 실험용)
        "opt2": compute_rr_features_opt2,  # 2 features (DAEAC 논문)
        # "opt3": compute_rr_features_opt3,  # 향후 확장용
    }

    if option not in functions:
        raise ValueError(f"Unknown RR option: '{option}'. Available: {list(functions.keys())}")

    return functions[option]


def compute_rr_features_opt1(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 360
) -> np.ndarray:
    """
    기존 실험용 RR features (7 features) - Cross-Attention 연구

    Features:
        [0] pre_rr         - Current RR interval (ms)
        [1] post_rr        - Next RR interval (ms)
        [2] local_rr       - Local mean RR (last 10 beats)
        [3] pre_div_post   - RR_i / RR_{i+1}
        [4] global_rr      - Global RR mean
        [5] pre_minus_global - RR_i - global_RR
        [6] pre_div_global - RR_i / global_RR

    Returns:
        features: (n_beats, 7) array
    """
    n_beats = len(r_peaks)

    # Convert to ms units
    ms_factor = 1000.0 / fs

    # Initialize RR arrays
    pre_rr = np.zeros(n_beats, dtype=np.float32)
    post_rr = np.zeros(n_beats, dtype=np.float32)
    local_rr = np.zeros(n_beats, dtype=np.float32)

    # Pre RR and Post RR 계산
    for i in range(n_beats):
        if i > 0:
            pre_rr[i] = (r_peaks[i] - r_peaks[i-1]) * ms_factor
        else:
            if n_beats > 1:
                pre_rr[i] = (r_peaks[1] - r_peaks[0]) * ms_factor
            else:
                pre_rr[i] = 800.0

        if i < n_beats - 1:
            post_rr[i] = (r_peaks[i+1] - r_peaks[i]) * ms_factor
        else:
            if n_beats > 1:
                post_rr[i] = (r_peaks[-1] - r_peaks[-2]) * ms_factor
            else:
                post_rr[i] = 800.0

    # Local statistics (last 10 beats)
    for i in range(n_beats):
        start_idx = max(0, i - 9)
        window = pre_rr[start_idx:i+1]
        if len(window) > 0:
            local_rr[i] = np.mean(window)
        else:
            local_rr[i] = pre_rr[i]

    # Global statistics
    valid_pre_rr = pre_rr[pre_rr > 50]

    if len(valid_pre_rr) > 1:
        global_rr_mean = np.mean(valid_pre_rr)
    else:
        global_rr_mean = 800.0

    # global_rr 배열 생성 (버그 수정: 실제 값 할당)
    global_rr = np.full(n_beats, global_rr_mean, dtype=np.float32)

    # Derived features
    epsilon = 1.0

    pre_div_post = pre_rr / np.maximum(post_rr, epsilon)
    pre_minus_global = pre_rr - global_rr
    pre_div_global = pre_rr / np.maximum(global_rr, epsilon)

    # Stack all features (n_beats, 7)
    all_features = np.stack([
        pre_rr,            # [0] Current RR interval
        post_rr,           # [1] Next RR interval
        local_rr,          # [2] Local mean RR (last 10 beats)
        pre_div_post,      # [3] RR_i / RR_{i+1}
        global_rr,         # [4] Global RR mean
        pre_minus_global,  # [5] RR_i - global_RR
        pre_div_global,    # [6] RR_i / global_RR
    ], axis=1).astype(np.float32)

    return all_features


def compute_rr_features_opt2(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 360
) -> np.ndarray:
    """
    DAEAC 논문 방식의 RR Feature 추출 (2 features)

    Reference:
        "Inter-patient ECG arrhythmia heartbeat classification based on
        unsupervised domain adaptation" (Wang et al., 2021)

    Features:
        [0] pre_rr_ratio      - 현재 pre-RR / 현재까지 모든 pre-RR의 평균
        [1] near_pre_rr_ratio - 현재 pre-RR / 최근 10개 pre-RR의 평균

    Note:
        논문에서는 이 2개의 scalar 값을 ECG segment와 같은 길이로 repeat하여
        3채널 입력 (ECG, pre_rr_ratio, near_pre_rr_ratio)으로 사용함

    Returns:
        features: (n_beats, 2) array
    """
    n_beats = len(r_peaks)

    # Initialize arrays
    pre_rr = np.zeros(n_beats, dtype=np.float32)
    pre_rr_ratio = np.zeros(n_beats, dtype=np.float32)
    near_pre_rr_ratio = np.zeros(n_beats, dtype=np.float32)

    # Step 1: Compute pre-RR intervals (in samples)
    for i in range(n_beats):
        if i > 0:
            pre_rr[i] = r_peaks[i] - r_peaks[i-1]
        else:
            # 첫 번째 beat: 다음 RR interval 사용
            if n_beats > 1:
                pre_rr[i] = r_peaks[1] - r_peaks[0]
            else:
                pre_rr[i] = fs  # default 1초 (360 samples at 360Hz)

    # Step 2: Compute RR ratios
    epsilon = 1.0  # 0으로 나누기 방지

    for i in range(n_beats):
        # pre_rr_ratio: 현재 pre-RR / 현재까지 모든 pre-RR의 평균
        if i > 0:
            avg_all_pre_rr = np.mean(pre_rr[:i+1])
            pre_rr_ratio[i] = pre_rr[i] / max(avg_all_pre_rr, epsilon)
        else:
            pre_rr_ratio[i] = 1.0  # 첫 번째 beat는 ratio = 1

        # near_pre_rr_ratio: 현재 pre-RR / 최근 10개 pre-RR의 평균
        start_idx = max(0, i - 9)  # 최근 10개 (현재 포함)
        if i > 0:
            near_avg = np.mean(pre_rr[start_idx:i+1])
            near_pre_rr_ratio[i] = pre_rr[i] / max(near_avg, epsilon)
        else:
            near_pre_rr_ratio[i] = 1.0

    # Stack features
    all_features = np.stack([
        pre_rr_ratio,       # [0] pre-RR ratio (global)
        near_pre_rr_ratio,  # [1] near-pre-RR ratio (local, last 10)
    ], axis=1).astype(np.float32)
    
    return all_features


# =============================================================================
# Beat Extraction - Default Style
# =============================================================================

def process_single_record(args) -> Tuple:
    """단일 레코드 처리 (기본 스타일)"""
    rec, base_path, valid_leads, out_len, patient_id = args

    data = []
    labels = []
    rr_feats = []
    pids = []
    indexes = []
    skipped = 0

    try:
        ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
        sig, meta = wfdb.rdsamp(os.path.join(base_path, rec))
    except Exception as e:
        print(f"Warning: {rec} read failed - {e}")
        return data, labels, rr_feats, pids, indexes, skipped

    fs = int(meta['fs'])
    sig_names = meta['sig_name']

    ch_idx = None
    for lead in valid_leads:
        if lead in sig_names:
            ch_idx = sig_names.index(lead)
            break
    if ch_idx is None:
        return data, labels, rr_feats, pids, indexes, skipped

    x = sig[:, ch_idx].astype(np.float32)
    x_filtered = remove_baseline_bandpass(x, fs=fs)

    r_peaks = ann.sample
    rr_func = get_rr_feature_function(RR_FEATURE_OPTION)
    rr_features = rr_func(x_filtered, r_peaks, fs)

    pre = int(round(360 * fs / 360.0))
    post = int(round(360 * fs / 360.0))

    for idx, (center, symbol) in enumerate(zip(ann.sample, ann.symbol)):
        grp = LABEL_GROUP_MAP.get(symbol, None)
        if grp is None or grp not in CLASSES:
            continue

        start = center - pre
        end = center + post
        if start < 0 or end > len(x_filtered):
            skipped += 1
            continue

        seg = x_filtered[start:end]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        seg_resampled = resample_to_len(seg, out_len)

        data.append(seg_resampled)
        labels.append(LABEL_TO_ID[grp])
        rr_feats.append(rr_features[idx])
        pids.append(patient_id)
        indexes.append(idx)

    return data, labels, rr_feats, pids, indexes, skipped


def extract_beats_and_rr_from_records(
    record_list: list,
    base_path: str,
    valid_leads: list,
    out_len: int,
    split_name: str,
    n_jobs: int = None
) -> tuple:
    """기본 스타일로 Beat 추출 (병렬 처리)"""
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}

    args = [
        (rec, base_path, valid_leads, out_len, patient_to_id[rec])
        for rec in record_list
    ]

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(delayed(process_single_record)(a) for a in args)

    all_data, all_labels, all_rr, all_pids, all_idx = [], [], [], [], []
    skipped_total = 0

    for data, labels, rr, pids, idxs, skipped in results:
        all_data.extend(data)
        all_labels.extend(labels)
        all_rr.extend(rr)
        all_pids.extend(pids)
        all_idx.extend(idxs)
        skipped_total += skipped

    print(f"{split_name} - Skipped: {skipped_total}, Extracted: {len(all_data)}")

    return (
        np.asarray(all_data, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_rr, dtype=np.float32),
        np.asarray(all_pids, dtype=np.int64),
        np.asarray(all_idx, dtype=np.int64),
    )


def extract_beats_and_rr_from_records_single(
    record_list: list,
    base_path: str,
    valid_leads: list,
    out_len: int,
    split_name: str
) -> tuple:
    """기본 스타일로 Beat 추출 (단일 스레드)"""
    all_data = []
    all_labels_id = []
    all_rr_features = []
    all_patient_ids = []
    all_indexes = []
    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}

    skipped = 0
    for rec in tqdm(record_list, desc=f"Extracting {split_name}"):
        try:
            ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
            sig, meta = wfdb.rdsamp(os.path.join(base_path, rec))
        except Exception as e:
            print(f"Warning: {rec} read failed - {e}")
            continue

        patient_id = patient_to_id[rec]
        fs = int(meta['fs'])
        sig_names = meta['sig_name']

        ch_idx = None
        for lead in valid_leads:
            if lead in sig_names:
                ch_idx = sig_names.index(lead)
                break

        if ch_idx is None:
            continue

        x = sig[:, ch_idx].astype(np.float32)
        x_filtered = remove_baseline_bandpass(x, fs=fs)

        r_peaks = ann.sample
        rr_func = get_rr_feature_function(RR_FEATURE_OPTION)
        rr_features = rr_func(x_filtered, r_peaks, fs)

        pre = int(round(360 * fs / 360.0))
        post = int(round(360 * fs / 360.0))

        for idx, (center, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            grp = LABEL_GROUP_MAP.get(symbol, None)

            if grp is None or grp not in CLASSES:
                continue

            start = center - pre
            end = center + post

            if start < 0 or end > len(x_filtered):
                skipped += 1
                continue

            seg = x_filtered[start:end]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            seg_resampled = resample_to_len(seg, out_len)

            all_data.append(seg_resampled)
            all_labels_id.append(LABEL_TO_ID[grp])
            all_rr_features.append(rr_features[idx])
            all_patient_ids.append(patient_id)
            all_indexes.append(idx)

    print(f"{split_name} - Skipped: {skipped}, Extracted: {len(all_data)}")

    return (np.array(all_data, dtype=np.float32),
            np.array(all_labels_id, dtype=np.int64),
            np.array(all_rr_features, dtype=np.float32),
            np.array(all_patient_ids, dtype=np.int64),
            np.array(all_indexes, dtype=np.int64))


# =============================================================================
# Beat Extraction - DAEAC Style
# =============================================================================

def _process_record_daeac(args):
    """
    DAEAC 논문 방식으로 단일 레코드 처리

    Segmentation:
        - Start: 0.14s after previous R-peak
        - End: 0.28s after current R-peak
    """
    rec, base_path, valid_leads, out_len, patient_id = args

    data = []
    labels = []
    rr_feats = []
    pids = []
    indexes = []
    skipped = 0

    try:
        ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
        sig, meta = wfdb.rdsamp(os.path.join(base_path, rec))
    except Exception as e:
        print(f"Warning: {rec} read failed - {e}")
        return data, labels, rr_feats, pids, indexes, skipped

    fs = int(meta['fs'])
    sig_names = meta['sig_name']

    ch_idx = None
    for lead in valid_leads:
        if lead in sig_names:
            ch_idx = sig_names.index(lead)
            break
    if ch_idx is None:
        return data, labels, rr_feats, pids, indexes, skipped

    x = sig[:, ch_idx].astype(np.float32)
    x_filtered = remove_baseline_bandpass(x, fs=fs)

    r_peaks = ann.sample

    # DAEAC 논문 RR feature 계산 (opt2)
    rr_features = compute_rr_features_opt2(x_filtered, r_peaks, fs)

    # 논문의 세그멘테이션 파라미터
    offset_after_prev = int(round(0.14 * fs))  # 0.14초
    offset_after_curr = int(round(0.28 * fs))  # 0.28초

    for idx in range(1, len(ann.sample)):  # 첫 번째 beat는 이전 R-peak가 없으므로 스킵
        symbol = ann.symbol[idx]
        grp = LABEL_GROUP_MAP.get(symbol, None)
        if grp is None or grp not in CLASSES:
            continue

        prev_r = r_peaks[idx - 1]
        curr_r = r_peaks[idx]

        # 세그먼트 범위: 이전 R-peak + 0.14s ~ 현재 R-peak + 0.28s
        start = prev_r + offset_after_prev
        end = curr_r + offset_after_curr

        if start < 0 or end > len(x_filtered) or start >= end:
            skipped += 1
            continue

        seg = x_filtered[start:end]

        # Z-score 정규화
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)

        # 논문: 360Hz로 리샘플링 후 길이 128로 고정
        seg_resampled = resample_to_len(seg, out_len)

        data.append(seg_resampled)
        labels.append(LABEL_TO_ID[grp])
        rr_feats.append(rr_features[idx])
        pids.append(patient_id)
        indexes.append(idx)

    return data, labels, rr_feats, pids, indexes, skipped


def extract_beats_daeac_style(
    record_list: list,
    base_path: str,
    valid_leads: list,
    out_len: int,
    split_name: str,
    n_jobs: int = None
) -> tuple:
    """
    DAEAC 논문 방식의 ECG Beat 추출

    Reference:
        "Inter-patient ECG arrhythmia heartbeat classification based on
        unsupervised domain adaptation" (Wang et al., 2021)

    Segmentation:
        - Start: 0.14s after previous R-peak
        - End: 0.28s after current R-peak
        - Resampled to fixed length (128 in paper)

    Returns:
        (data, labels, rr_features, patient_ids, sample_ids)
        - data: (n_samples, out_len) ECG segments
        - rr_features: (n_samples, 2) [pre_rr_ratio, near_pre_rr_ratio]
    """
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}

    args = [
        (rec, base_path, valid_leads, out_len, patient_to_id[rec])
        for rec in record_list
    ]

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(delayed(_process_record_daeac)(a) for a in args)

    all_data, all_labels, all_rr, all_pids, all_idx = [], [], [], [], []
    skipped_total = 0

    for data, labels, rr, pids, idxs, skipped in results:
        all_data.extend(data)
        all_labels.extend(labels)
        all_rr.extend(rr)
        all_pids.extend(pids)
        all_idx.extend(idxs)
        skipped_total += skipped

    print(f"{split_name} (DAEAC style) - Skipped: {skipped_total}, Extracted: {len(all_data)}")

    # Ensure all_rr is properly shaped as (n_samples, 2)
    # Convert list of 1D arrays to 2D array explicitly
    if len(all_rr) > 0:
        rr_array = np.stack(all_rr, axis=0).astype(np.float32)
    else:
        rr_array = np.array([], dtype=np.float32).reshape(0, 2)
    
    # Validate shape
    if rr_array.ndim != 2:
        raise ValueError(f"RR features should be 2D array (n_samples, 2), got {rr_array.ndim}D with shape {rr_array.shape}")
    if rr_array.shape[1] != 2:
        raise ValueError(f"RR features should have 2 columns, got shape {rr_array.shape}")

    return (
        np.asarray(all_data, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
        rr_array,
        np.asarray(all_pids, dtype=np.int64),
        np.asarray(all_idx, dtype=np.int64),
    )


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Class-balanced Focal Loss"""

    def __init__(self, alpha=0.25, gamma=2.0, beta=0.999, class_counts=None,
                 device='cuda', reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.device = device

        if class_counts is not None:
            import torch
            if isinstance(class_counts, list):
                class_counts = torch.tensor(class_counts, dtype=torch.float32)

            # effective number of samples
            effective_num = 1.0 - torch.pow(self.beta, class_counts)
            weights = (1.0 - self.beta) / effective_num

            # optional normalization (논문에서 권장)
            weights = weights / weights.sum() * len(class_counts)

            self.weights = weights.to(device)
            print(f"CB-Focal weights: {self.weights.cpu().numpy()}")
        else:
            self.weights = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_term * log_pt + torch.pow(1.0 - pt, self.gamma + 1)

        if self.weights is not None:
            loss = loss * self.weights[targets]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =============================================================================
# Dataset Caching System
# =============================================================================

def _compute_cache_hash(record_list: list, out_len: int, valid_leads: list) -> str:
    """record 리스트, out_len, leads, RR option을 기반으로 고유 해시 생성"""
    cache_key = {
        'records': sorted(record_list),
        'out_len': out_len,
        'valid_leads': valid_leads,
        'rr_option': RR_FEATURE_OPTION,
    }
    key_str = json.dumps(cache_key, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _get_cache_paths(split_name: str, cache_hash: str, cache_dir: str = CACHE_DIR) -> dict:
    """캐시 파일 경로들 반환"""
    os.makedirs(cache_dir, exist_ok=True)
    prefix = f"{split_name}_{cache_hash}"
    return {
        'data': os.path.join(cache_dir, f"{prefix}_data.npy"),
        'labels': os.path.join(cache_dir, f"{prefix}_labels.npy"),
        'rr': os.path.join(cache_dir, f"{prefix}_rr.npy"),
        'patient_ids': os.path.join(cache_dir, f"{prefix}_patient_ids.npy"),
        'sample_ids': os.path.join(cache_dir, f"{prefix}_sample_ids.npy"),
        'meta': os.path.join(cache_dir, f"{prefix}_meta.json"),
    }


def _is_cache_valid(cache_paths: dict, record_list: list, out_len: int) -> bool:
    """캐시 유효성 검사"""
    # 파일 존재 확인
    for key, path in cache_paths.items():
        if not os.path.exists(path):
            return False

    try:
        # 메타데이터 로드 및 비교
        with open(cache_paths['meta'], 'r') as f:
            meta = json.load(f)

        # record 리스트 일치 확인
        if sorted(meta.get('records', [])) != sorted(record_list):
            print("  Cache invalid: record list mismatch")
            return False

        # out_len 확인
        if meta.get('out_len') != out_len:
            print("  Cache invalid: out_len mismatch")
            return False

        # 샘플 shape 검증
        data_sample = np.load(cache_paths['data'], mmap_mode='r')
        if len(data_sample) == 0:
            print("  Cache invalid: empty data")
            return False

        if data_sample.shape[1] != out_len:
            print(f"  Cache invalid: signal length mismatch ({data_sample.shape[1]} vs {out_len})")
            return False

        # RR feature dimension 확인
        rr_sample = np.load(cache_paths['rr'], mmap_mode='r')
        if len(rr_sample) > 0:
            # For DAEAC style, rr should be 2D with 2 columns
            expected_rr_dim = meta.get('rr_dim', 7)
            if rr_sample.ndim != 2:
                print(f"  Cache invalid: RR array should be 2D, got {rr_sample.ndim}D")
                return False
            if rr_sample.shape[1] != expected_rr_dim:
                print(f"  Cache invalid: RR dimension mismatch ({rr_sample.shape[1]} vs {expected_rr_dim})")
                return False

        return True

    except Exception as e:
        print(f"  Cache validation error: {e}")
        return False


def _save_cache(cache_paths: dict, data: tuple, record_list: list, out_len: int, rr_dim: int):
    """데이터셋을 numpy 파일로 저장"""
    data_arr, labels_arr, rr_arr, patient_ids, sample_ids = data

    np.save(cache_paths['data'], data_arr)
    np.save(cache_paths['labels'], labels_arr)
    np.save(cache_paths['rr'], rr_arr)
    np.save(cache_paths['patient_ids'], patient_ids)
    np.save(cache_paths['sample_ids'], sample_ids)

    # 메타데이터 저장
    meta = {
        'records': sorted(record_list),
        'out_len': out_len,
        'rr_dim': rr_dim,
        'n_samples': len(data_arr),
        'data_shape': list(data_arr.shape),
        'rr_shape': list(rr_arr.shape),
        'label_distribution': {int(k): int(v) for k, v in zip(*np.unique(labels_arr, return_counts=True))},
    }

    with open(cache_paths['meta'], 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Cache saved: {len(data_arr)} samples")


def _load_cache(cache_paths: dict) -> tuple:
    """캐시에서 데이터 로드"""
    data = np.load(cache_paths['data'])
    labels = np.load(cache_paths['labels'])
    rr = np.load(cache_paths['rr'])
    patient_ids = np.load(cache_paths['patient_ids'])
    sample_ids = np.load(cache_paths['sample_ids'])

    return data, labels, rr, patient_ids, sample_ids


def load_or_extract_data(
    record_list: list,
    base_path: str,
    valid_leads: list,
    out_len: int,
    split_name: str,
    n_jobs: int = None,
    cache_dir: str = CACHE_DIR,
    force_reprocess: bool = False,
    extraction_style: str = "default"
) -> tuple:
    """
    캐시 시스템을 사용한 데이터 로드/추출

    Args:
        record_list: 레코드 ID 리스트
        base_path: 데이터 경로
        valid_leads: 유효한 리드 리스트
        out_len: 출력 시그널 길이
        split_name: 데이터셋 이름 (Train/Valid/Test)
        n_jobs: 병렬 처리 작업 수
        cache_dir: 캐시 저장 디렉토리
        force_reprocess: True면 캐시 무시하고 재처리
        extraction_style: "default" 또는 "daeac" (논문 스타일)

    Returns:
        (data, labels, rr_features, patient_ids, sample_ids)
    """
    # extraction_style을 캐시 해시에 포함
    cache_hash = _compute_cache_hash(record_list, out_len, valid_leads)
    if extraction_style == "daeac":
        cache_hash = cache_hash + "_daeac"

    cache_paths = _get_cache_paths(split_name, cache_hash, cache_dir)

    print(f"\n[{split_name}] Cache check (hash: {cache_hash}, style: {extraction_style})")

    # 캐시 유효성 검사
    if not force_reprocess and _is_cache_valid(cache_paths, record_list, out_len):
        print(f"  Loading from cache...")
        data, labels, rr, patient_ids, sample_ids = _load_cache(cache_paths)
        print(f"  Loaded {len(data)} samples from cache")
        print(f"  RR features shape: {rr.shape if hasattr(rr, 'shape') else type(rr)}")
        return data, labels, rr, patient_ids, sample_ids

    # 캐시 없거나 유효하지 않음 -> 전처리 실행
    print(f"  Cache not found or invalid. Processing...")

    if extraction_style == "daeac":
        # DAEAC 논문 스타일 추출
        data, labels, rr, patient_ids, sample_ids = extract_beats_daeac_style(
            record_list=record_list,
            base_path=base_path,
            valid_leads=valid_leads,
            out_len=out_len,
            split_name=split_name,
            n_jobs=n_jobs
        )
    else:
        # 기본 스타일 추출
        data, labels, rr, patient_ids, sample_ids = extract_beats_and_rr_from_records(
            record_list=record_list,
            base_path=base_path,
            valid_leads=valid_leads,
            out_len=out_len,
            split_name=split_name,
            n_jobs=n_jobs
        )

    # Debug: Print extracted RR shape
    print(f"  Extracted RR features shape: {rr.shape if hasattr(rr, 'shape') else type(rr)}")
    
    # 캐시 저장
    if len(data) > 0:
        rr_dim = rr.shape[1] if len(rr.shape) > 1 else 0
        _save_cache(cache_paths, (data, labels, rr, patient_ids, sample_ids),
                   record_list, out_len, rr_dim)

    return data, labels, rr, patient_ids, sample_ids
