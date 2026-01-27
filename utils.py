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
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
    """RR feature 옵션에 따라 적절한 함수 반환"""
    functions = {
        "opt1": compute_rr_features_opt1,  # 7 features
        "opt2": compute_rr_features_opt2,  # 2 features (DAEAC)
    }

    if option not in functions:
        raise ValueError(f"Unknown RR option: '{option}'. Available: {list(functions.keys())}")

    return functions[option]


def compute_rr_features_opt1(ecg_signal: np.ndarray, r_peaks: np.ndarray, fs: int = 360) -> np.ndarray:
    """기존 실험용 RR features (7 features)"""
    n_beats = len(r_peaks)
    ms_factor = 1000.0 / fs

    pre_rr = np.zeros(n_beats, dtype=np.float32)
    post_rr = np.zeros(n_beats, dtype=np.float32)
    local_rr = np.zeros(n_beats, dtype=np.float32)

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

    for i in range(n_beats):
        start_idx = max(0, i - 9)
        window = pre_rr[start_idx:i+1]
        if len(window) > 0:
            local_rr[i] = np.mean(window)
        else:
            local_rr[i] = pre_rr[i]

    valid_pre_rr = pre_rr[pre_rr > 50]
    if len(valid_pre_rr) > 1:
        global_rr_mean = np.mean(valid_pre_rr)
    else:
        global_rr_mean = 800.0

    global_rr = np.full(n_beats, global_rr_mean, dtype=np.float32)

    epsilon = 1.0
    pre_div_post = pre_rr / np.maximum(post_rr, epsilon)
    pre_minus_global = pre_rr - global_rr
    pre_div_global = pre_rr / np.maximum(global_rr, epsilon)

    all_features = np.stack([
        pre_rr, post_rr, local_rr, pre_div_post,
        global_rr, pre_minus_global, pre_div_global
    ], axis=1).astype(np.float32)

    return all_features


def compute_rr_features_opt2(ecg_signal: np.ndarray, r_peaks: np.ndarray, fs: int = 360) -> np.ndarray:
    """DAEAC 논문 방식의 RR Feature 추출 (2 features)"""
    n_beats = len(r_peaks)
    pre_rr = np.zeros(n_beats, dtype=np.float32)
    pre_rr_ratio = np.zeros(n_beats, dtype=np.float32)
    near_pre_rr_ratio = np.zeros(n_beats, dtype=np.float32)

    for i in range(n_beats):
        if i > 0:
            pre_rr[i] = r_peaks[i] - r_peaks[i-1]
        else:
            if n_beats > 1:
                pre_rr[i] = r_peaks[1] - r_peaks[0]
            else:
                pre_rr[i] = fs

    epsilon = 1.0
    for i in range(n_beats):
        if i > 0:
            avg_all_pre_rr = np.mean(pre_rr[:i+1])
            pre_rr_ratio[i] = pre_rr[i] / max(avg_all_pre_rr, epsilon)
        else:
            pre_rr_ratio[i] = 1.0

        start_idx = max(0, i - 9)
        if i > 0:
            near_avg = np.mean(pre_rr[start_idx:i+1])
            near_pre_rr_ratio[i] = pre_rr[i] / max(near_avg, epsilon)
        else:
            near_pre_rr_ratio[i] = 1.0

    all_features = np.stack([pre_rr_ratio, near_pre_rr_ratio], axis=1).astype(np.float32)
    return all_features


# =============================================================================
# Beat Extraction - Default Style (Single Threaded)
# =============================================================================

def extract_beats_and_rr_from_records(
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

    # tqdm을 사용한 진행률 표시
    for rec in tqdm(record_list, desc=f"Extracting {split_name} (Default)"):
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
        # 기본 스타일은 항상 opt1 사용
        rr_func = get_rr_feature_function("opt1")
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
# Beat Extraction - DAEAC Style (Single Threaded)
# =============================================================================

def extract_beats_daeac_style(
    record_list: list,
    base_path: str,
    valid_leads: list,
    out_len: int,
    split_name: str
) -> tuple:
    """DAEAC 논문 방식의 Beat 추출 (단일 스레드)"""
    all_data = []
    all_labels = []
    all_rr = []
    all_pids = []
    all_indexes = []
    
    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}
    skipped = 0

    for rec in tqdm(record_list, desc=f"Extracting {split_name} (DAEAC)"):
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

        # DAEAC 논문 RR feature (opt2)
        rr_features = compute_rr_features_opt2(x_filtered, r_peaks, fs)

        offset_after_prev = int(round(0.14 * fs))
        offset_after_curr = int(round(0.28 * fs))

        for idx in range(1, len(ann.sample)):
            symbol = ann.symbol[idx]
            grp = LABEL_GROUP_MAP.get(symbol, None)
            
            if grp is None or grp not in CLASSES:
                continue

            prev_r = r_peaks[idx - 1]
            curr_r = r_peaks[idx]

            start = prev_r + offset_after_prev
            end = curr_r + offset_after_curr

            if start < 0 or end > len(x_filtered) or start >= end:
                skipped += 1
                continue

            seg = x_filtered[start:end]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            seg_resampled = resample_to_len(seg, out_len)

            all_data.append(seg_resampled)
            all_labels.append(LABEL_TO_ID[grp])
            all_rr.append(rr_features[idx])
            all_pids.append(patient_id)
            all_indexes.append(idx)

    print(f"{split_name} (DAEAC style) - Skipped: {skipped}, Extracted: {len(all_data)}")

    if len(all_rr) > 0:
        rr_array = np.stack(all_rr, axis=0).astype(np.float32)
    else:
        rr_array = np.array([], dtype=np.float32).reshape(0, 2)

    return (
        np.asarray(all_data, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
        rr_array,
        np.asarray(all_pids, dtype=np.int64),
        np.asarray(all_indexes, dtype=np.int64),
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

            effective_num = 1.0 - torch.pow(self.beta, class_counts)
            weights = (1.0 - self.beta) / effective_num
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

def _compute_cache_hash(record_list: list, out_len: int, valid_leads: list, extraction_style: str = "default") -> str:
    """record 리스트, out_len, leads, RR option을 기반으로 고유 해시 생성"""
    if extraction_style == "default":
        rr_option = "opt1"
    else:
        rr_option = RR_FEATURE_OPTION
    
    cache_key = {
        'records': sorted(record_list),
        'out_len': out_len,
        'valid_leads': valid_leads,
        'rr_option': rr_option,
        'extraction_style': extraction_style,
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


def _is_cache_valid(cache_paths: dict, record_list: list, out_len: int, extraction_style: str = "default") -> bool:
    """캐시 유효성 검사"""
    for key, path in cache_paths.items():
        if not os.path.exists(path):
            return False

    try:
        with open(cache_paths['meta'], 'r') as f:
            meta = json.load(f)

        cached_style = meta.get('extraction_style')
        if cached_style is None:
            print(f"  Cache invalid: extraction_style not found (old format)")
            return False
        if cached_style != extraction_style:
            print(f"  Cache invalid: style mismatch ({cached_style} vs {extraction_style})")
            return False

        if sorted(meta.get('records', [])) != sorted(record_list):
            print("  Cache invalid: record list mismatch")
            return False

        if meta.get('out_len') != out_len:
            print("  Cache invalid: out_len mismatch")
            return False

        data_sample = np.load(cache_paths['data'], mmap_mode='r')
        if len(data_sample) == 0:
            print("  Cache invalid: empty data")
            return False

        if data_sample.shape[1] != out_len:
            print(f"  Cache invalid: length mismatch ({data_sample.shape[1]} vs {out_len})")
            return False

        rr_sample = np.load(cache_paths['rr'], mmap_mode='r')
        if len(rr_sample) > 0:
            if extraction_style == "default":
                expected_rr_dim = 7
            else:
                expected_rr_dim = 2
            
            if rr_sample.ndim != 2:
                print(f"  Cache invalid: RR not 2D")
                return False
            if rr_sample.shape[1] != expected_rr_dim:
                print(f"  Cache invalid: RR dim mismatch ({rr_sample.shape[1]} vs {expected_rr_dim})")
                return False

        return True

    except Exception as e:
        print(f"  Cache validation error: {e}")
        return False


def _save_cache(cache_paths: dict, data: tuple, record_list: list, out_len: int, rr_dim: int, extraction_style: str = "default"):
    """데이터셋을 numpy 파일로 저장"""
    data_arr, labels_arr, rr_arr, patient_ids, sample_ids = data

    np.save(cache_paths['data'], data_arr)
    np.save(cache_paths['labels'], labels_arr)
    np.save(cache_paths['rr'], rr_arr)
    np.save(cache_paths['patient_ids'], patient_ids)
    np.save(cache_paths['sample_ids'], sample_ids)

    meta = {
        'records': sorted(record_list),
        'out_len': out_len,
        'rr_dim': rr_dim,
        'extraction_style': extraction_style,
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
    n_jobs: int = None,  # 호환성을 위해 남겨둠 (사용 안함)
    cache_dir: str = CACHE_DIR,
    force_reprocess: bool = False,
    extraction_style: str = "default"
) -> tuple:
    """캐시 시스템을 사용한 데이터 로드/추출 (Single Threaded)"""
    cache_hash = _compute_cache_hash(record_list, out_len, valid_leads, extraction_style)
    cache_paths = _get_cache_paths(split_name, cache_hash, cache_dir)

    print(f"\n[{split_name}] Cache check (hash: {cache_hash}, style: {extraction_style})")

    if not force_reprocess and _is_cache_valid(cache_paths, record_list, out_len, extraction_style):
        print(f"  Loading from cache...")
        data, labels, rr, patient_ids, sample_ids = _load_cache(cache_paths)
        print(f"  Loaded {len(data)} samples from cache")
        return data, labels, rr, patient_ids, sample_ids

    print(f"  Cache not found or invalid. Processing...")

    if extraction_style == "daeac":
        data, labels, rr, patient_ids, sample_ids = extract_beats_daeac_style(
            record_list=record_list,
            base_path=base_path,
            valid_leads=valid_leads,
            out_len=out_len,
            split_name=split_name
        )
    else:
        data, labels, rr, patient_ids, sample_ids = extract_beats_and_rr_from_records(
            record_list=record_list,
            base_path=base_path,
            valid_leads=valid_leads,
            out_len=out_len,
            split_name=split_name
        )

    print(f"  Extracted RR features shape: {rr.shape if hasattr(rr, 'shape') else type(rr)}")
    
    if len(data) > 0:
        rr_dim = rr.shape[1] if len(rr.shape) > 1 else 0
        _save_cache(cache_paths, (data, labels, rr, patient_ids, sample_ids),
                    record_list, out_len, rr_dim, extraction_style)

    return data, labels, rr, patient_ids, sample_ids