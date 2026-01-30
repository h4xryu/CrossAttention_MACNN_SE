"""
Refactored ResU Experiment Script (Final Integration)
- Structure: Based on main_autoexp.py (Excel logging, Experiment Grid, Modular run)
- Implementation: Uses robust data processing, caching, and RR extraction from provided utils.
"""

import os
import sys
import time
import copy
import json
import hashlib
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score

import wfdb
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt
from tqdm import tqdm

# =============================================================================
# [설정] Config & Constants
# =============================================================================

# 경로 설정 (사용자 환경에 맞게 수정 필요)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'mit-bih-arrhythmia-database-1.0.0')
OUTPUT_DIR = "./autoexp_results_ResU/"
CACHE_DIR = './dataset_cache'

# 데이터 관련 상수
CLASSES = ['N', 'S', 'V', 'F']
LABEL_GROUP_MAP = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', 
                   'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S', 
                   'V': 'V', 'E': 'V', 'F': 'F'}
LABEL_TO_ID = {'N': 0, 'S': 1, 'V': 2, 'F': 3}
RR_FEATURE_OPTION = "opt1"  # Default for cache hashing

# 하이퍼파라미터
BATCH_SIZE = 1024
EPOCHS = 75
LR = 0.0001
WEIGHT_DECAY = 1e-2
SEED = 1234
OUT_LEN = 720
VALID_LEADS = ['MLII', 'II'] # 리드 설정

# ResU 모델 설정
BASE_MODEL_CONFIG = {
    'in_channels': 1,
    'out_ch': 180,
    'mid_ch': 30,
    'num_heads': 1,
    'n_rr_opt1': 7,  # Script 2의 compute_rr_features_opt1 (7 features)
    'n_rr_opt3': 2,  # Script 2의 compute_rr_features_opt2 (2 features)
    'n_channels_opt3': 3,
}

# 실험 그리드
EXPERIMENT_GRID = [
    # (실험명, 모델타입, 입력모드)
    # opt1 (7 features)
    {'name': 'ResU_opt1_baseline', 'model_type': 'baseline', 'input_mode': 'opt1'},
    {'name': 'ResU_opt1_naive',    'model_type': 'naive_concat', 'input_mode': 'opt1'},
    {'name': 'ResU_opt1_cross',    'model_type': 'cross_attention', 'input_mode': 'opt1'},
    # opt3 (DAEAC style, 2 features)
    {'name': 'ResU_opt3_baseline', 'model_type': 'baseline', 'input_mode': 'opt3'},
    {'name': 'ResU_opt3_naive',    'model_type': 'naive_concat', 'input_mode': 'opt3'},
    {'name': 'ResU_opt3_cross',    'model_type': 'cross_attention', 'input_mode': 'opt3'},
]

# 데이터 분할
DS1_TRAIN = ['101', '106', '108', '109', '112', '115', '116', '118', '119', '122', '201', '203', '209', '215', '223', '230']
DS1_VALID = ['114', '124', '205', '207', '220', '208']
DS2_TEST = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

# =============================================================================
# [Import] External Modules & Mocking
# =============================================================================
try:
    from src.utils import ExcelResultWriter, CumulativeExcelWriter
except ImportError:
    print("Warning: src.utils not found. Using Mock Writers.")
    class ExcelResultWriter:
        def __init__(self, *args, **kwargs): pass
        def write_metrics(self, *args, **kwargs): pass
        def write_confusion_matrix(self, *args, **kwargs): pass
    class CumulativeExcelWriter:
        def __init__(self, *args, **kwargs): pass
        def get_record_count(self): return 0
        def append_result(self, *args, **kwargs): pass
        def append_confusion_matrix(self, *args, **kwargs): pass

# ResU 모델 임포트
sys.path.append(os.path.join(SCRIPT_DIR, 'src', 'models'))
try:
    from ResU import get_resu_model
except ImportError:
    print("Error: ResU model not found in src/models/ResU.py. Please ensure the file exists.")
    # Mock for testing if model file is missing
    # sys.exit(1) 
    def get_resu_model(**kwargs): return nn.Linear(720, 4) 

# =============================================================================
# [Utils] Signal Processing & RR Features (From Script 2)
# =============================================================================

def set_seed(seed: int, fully_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def resample_to_len(ts: np.ndarray, out_len: int) -> np.ndarray:
    if len(ts) == out_len: return ts.astype(np.float32)
    if len(ts) < 2: return np.pad(ts.astype(np.float32), (0, max(0, out_len - len(ts))), mode='edge')[:out_len]
    x_old = np.linspace(0.0, 1.0, num=len(ts))
    x_new = np.linspace(0.0, 1.0, num=out_len)
    return interp1d(x_old, ts, kind='linear')(x_new).astype(np.float32)

def remove_baseline_bandpass(signal: np.ndarray, fs: int = 360, lowcut: float = 0.1, highcut: float = 100.0, order: int = 256) -> np.ndarray:
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    fir_coeff = firwin(order + 1, [low, high], pass_zero=False)
    return filtfilt(fir_coeff, 1.0, signal)

# --- RR Feature Functions ---

def compute_rr_features_opt1(ecg_signal: np.ndarray, r_peaks: np.ndarray, fs: int = 360) -> np.ndarray:
    """7-Dim Full Ratio Strategy (Robust)"""
    n_beats = len(r_peaks)
    ms_factor = 1000.0 / fs

    pre_rr = np.zeros(n_beats, dtype=np.float32)
    post_rr = np.zeros(n_beats, dtype=np.float32)
    
    for i in range(n_beats):
        if i > 0: pre_rr[i] = (r_peaks[i] - r_peaks[i-1]) * ms_factor
        else: pre_rr[i] = 800.0 if n_beats <= 1 else (r_peaks[1] - r_peaks[0]) * ms_factor
        
        if i < n_beats - 1: post_rr[i] = (r_peaks[i+1] - r_peaks[i]) * ms_factor
        else: post_rr[i] = 800.0 if n_beats <= 1 else (r_peaks[-1] - r_peaks[-2]) * ms_factor

    local_rr = np.zeros(n_beats, dtype=np.float32)
    for i in range(n_beats):
        start_idx = max(0, i - 9)
        window = pre_rr[start_idx:i+1]
        local_rr[i] = np.mean(window) if len(window) > 0 else pre_rr[i]

    valid_pre_rr = pre_rr[pre_rr > 50]
    global_rr_val = np.mean(valid_pre_rr) if len(valid_pre_rr) > 1 else 800.0
    global_rr = np.full(n_beats, global_rr_val, dtype=np.float32)

    epsilon = 1e-8
    feat0 = pre_rr / (local_rr + epsilon)
    feat1 = pre_rr / (post_rr + epsilon)
    feat2 = np.ones(n_beats, dtype=np.float32)
    feat2[1:] = pre_rr[1:] / (pre_rr[:-1] + epsilon)
    feat3 = post_rr / (local_rr + epsilon)
    feat4 = local_rr / (global_rr + epsilon)
    feat5 = np.ones(n_beats, dtype=np.float32)
    feat5[2:] = pre_rr[2:] / (pre_rr[:-2] + epsilon)
    feat6 = np.ones(n_beats, dtype=np.float32)
    feat6[:-1] = post_rr[:-1] / (post_rr[1:] + epsilon)

    return np.stack([feat0, feat1, feat2, feat3, feat4, feat5, feat6], axis=1).astype(np.float32)

def compute_rr_features_opt2(ecg_signal: np.ndarray, r_peaks: np.ndarray, fs: int = 360) -> np.ndarray:
    """DAEAC Style (2 features)"""
    n_beats = len(r_peaks)
    pre_rr = np.zeros(n_beats, dtype=np.float32)
    pre_rr_ratio = np.zeros(n_beats, dtype=np.float32)
    near_pre_rr_ratio = np.zeros(n_beats, dtype=np.float32)

    for i in range(n_beats):
        if i > 0: pre_rr[i] = r_peaks[i] - r_peaks[i-1]
        else: pre_rr[i] = r_peaks[1] - r_peaks[0] if n_beats > 1 else fs

    epsilon = 1.0
    for i in range(n_beats):
        if i > 0:
            avg_all_pre_rr = np.mean(pre_rr[:i+1])
            pre_rr_ratio[i] = pre_rr[i] / max(avg_all_pre_rr, epsilon)
        else: pre_rr_ratio[i] = 1.0

        start_idx = max(0, i - 9)
        if i > 0:
            near_avg = np.mean(pre_rr[start_idx:i+1])
            near_pre_rr_ratio[i] = pre_rr[i] / max(near_avg, epsilon)
        else: near_pre_rr_ratio[i] = 1.0

    return np.stack([pre_rr_ratio, near_pre_rr_ratio], axis=1).astype(np.float32)

# =============================================================================
# [Data Extraction] Beat Extraction & Caching (From Script 2)
# =============================================================================

def extract_beats_and_rr_from_records(record_list, base_path, valid_leads, out_len, split_name):
    """Default style (Opt1)"""
    all_data, all_labels_id, all_rr_features, all_patient_ids, all_indexes = [], [], [], [], []
    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}
    skipped = 0

    for rec in tqdm(record_list, desc=f"Extracting {split_name} (Opt1/Default)"):
        try:
            ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
            sig, meta = wfdb.rdsamp(os.path.join(base_path, rec))
        except Exception as e:
            print(f"Warning: {rec} read failed - {e}"); continue

        patient_id = patient_to_id[rec]
        fs = int(meta['fs'])
        sig_names = meta['sig_name']
        ch_idx = next((sig_names.index(l) for l in valid_leads if l in sig_names), None)
        if ch_idx is None: continue

        x = sig[:, ch_idx].astype(np.float32)
        x_filtered = remove_baseline_bandpass(x, fs=fs)
        r_peaks = ann.sample
        rr_features = compute_rr_features_opt1(x_filtered, r_peaks, fs) # Use Opt1
        
        pre, post = int(round(fs)), int(round(fs)) # 1 sec window

        for idx, (center, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            grp = LABEL_GROUP_MAP.get(symbol, None)
            if grp is None or grp not in CLASSES: continue
            
            start, end = center - pre, center + post
            if start < 0 or end > len(x_filtered):
                skipped += 1; continue
            
            seg = x_filtered[start:end]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            seg_resampled = resample_to_len(seg, out_len)

            all_data.append(seg_resampled)
            all_labels_id.append(LABEL_TO_ID[grp])
            all_rr_features.append(rr_features[idx])
            all_patient_ids.append(patient_id)
            all_indexes.append(idx)

    return (np.array(all_data, dtype=np.float32), np.array(all_labels_id, dtype=np.int64),
            np.array(all_rr_features, dtype=np.float32), np.array(all_patient_ids, dtype=np.int64),
            np.array(all_indexes, dtype=np.int64))

def extract_beats_daeac_style(record_list, base_path, valid_leads, out_len, split_name):
    """DAEAC style (Opt3)"""
    all_data, all_labels, all_rr, all_pids, all_indexes = [], [], [], [], []
    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}
    skipped = 0

    for rec in tqdm(record_list, desc=f"Extracting {split_name} (Opt3/DAEAC)"):
        try:
            ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
            sig, meta = wfdb.rdsamp(os.path.join(base_path, rec))
        except Exception: continue

        patient_id = patient_to_id[rec]
        fs = int(meta['fs'])
        sig_names = meta['sig_name']
        ch_idx = next((sig_names.index(l) for l in valid_leads if l in sig_names), None)
        if ch_idx is None: continue

        x = sig[:, ch_idx].astype(np.float32)
        x_filtered = remove_baseline_bandpass(x, fs=fs)
        r_peaks = ann.sample
        rr_features = compute_rr_features_opt2(x_filtered, r_peaks, fs) # Use Opt2

        offset_prev, offset_curr = int(round(0.14 * fs)), int(round(0.28 * fs))

        for idx in range(1, len(ann.sample)):
            symbol = ann.symbol[idx]
            grp = LABEL_GROUP_MAP.get(symbol, None)
            if grp is None or grp not in CLASSES: continue

            prev_r, curr_r = r_peaks[idx - 1], r_peaks[idx]
            start, end = prev_r + offset_prev, curr_r + offset_curr

            if start < 0 or end > len(x_filtered) or start >= end:
                skipped += 1; continue

            seg = x_filtered[start:end]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            seg_resampled = resample_to_len(seg, out_len)

            all_data.append(seg_resampled)
            all_labels.append(LABEL_TO_ID[grp])
            all_rr.append(rr_features[idx])
            all_pids.append(patient_id)
            all_indexes.append(idx)

    rr_array = np.stack(all_rr, axis=0).astype(np.float32) if all_rr else np.empty((0,2), dtype=np.float32)
    return (np.asarray(all_data, dtype=np.float32), np.asarray(all_labels, dtype=np.int64),
            rr_array, np.asarray(all_pids, dtype=np.int64), np.asarray(all_indexes, dtype=np.int64))

# --- Caching System ---

def _compute_cache_hash(record_list, out_len, valid_leads, extraction_style):
    rr_option = "opt1" if extraction_style == "default" else "opt2"
    key_str = json.dumps({'records': sorted(record_list), 'out_len': out_len, 
                          'valid_leads': valid_leads, 'rr_option': rr_option, 
                          'extraction_style': extraction_style}, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]

def load_or_extract_data(record_list, base_path, valid_leads, out_len, split_name, cache_dir=CACHE_DIR, extraction_style="default"):
    cache_hash = _compute_cache_hash(record_list, out_len, valid_leads, extraction_style)
    os.makedirs(cache_dir, exist_ok=True)
    prefix = os.path.join(cache_dir, f"{split_name}_{cache_hash}")
    paths = {k: f"{prefix}_{k}.npy" for k in ['data', 'labels', 'rr', 'patient_ids', 'sample_ids']}
    paths['meta'] = f"{prefix}_meta.json"

    # Check cache
    if all(os.path.exists(p) for p in paths.values()):
        try:
            with open(paths['meta'], 'r') as f: meta = json.load(f)
            if meta.get('extraction_style') == extraction_style and meta.get('out_len') == out_len:
                print(f"[{split_name}] Loading from cache ({cache_hash})...")
                return (np.load(paths['data']), np.load(paths['labels']), np.load(paths['rr']), 
                        np.load(paths['patient_ids']), np.load(paths['sample_ids']))
        except Exception: pass

    # Process
    print(f"[{split_name}] Cache miss. Extracting ({extraction_style})...")
    if extraction_style == "daeac":
        data = extract_beats_daeac_style(record_list, base_path, valid_leads, out_len, split_name)
    else:
        data = extract_beats_and_rr_from_records(record_list, base_path, valid_leads, out_len, split_name)

    # Save cache
    if len(data[0]) > 0:
        np.save(paths['data'], data[0]); np.save(paths['labels'], data[1])
        np.save(paths['rr'], data[2]); np.save(paths['patient_ids'], data[3])
        np.save(paths['sample_ids'], data[4])
        with open(paths['meta'], 'w') as f:
            json.dump({'extraction_style': extraction_style, 'out_len': out_len, 'n_samples': len(data[0])}, f)
    
    return data

# =============================================================================
# [Datasets]
# =============================================================================

class ECGDataset_opt1(Dataset):
    def __init__(self, ecg_data, labels, rr_features, patient_ids, sample_ids):
        self.ecg_data = torch.FloatTensor(ecg_data)
        self.labels = torch.LongTensor(labels)
        self.rr_features = torch.FloatTensor(rr_features)
        self.patient_ids = torch.LongTensor(patient_ids)
        self.sample_ids = sample_ids
        if self.ecg_data.ndim == 2: # (N, L) -> (N, 1, L)
            self.ecg_data = self.ecg_data.unsqueeze(1)
            
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (self.ecg_data[idx], self.rr_features[idx], self.labels[idx], self.patient_ids[idx], idx)

class ECGDataset_opt3(Dataset):
    def __init__(self, ecg_data, labels, rr_features, patient_ids, sample_ids):
        self.ecg_data = ecg_data # numpy for easier concat
        self.labels = torch.LongTensor(labels)
        self.rr_features = rr_features
        self.patient_ids = torch.LongTensor(patient_ids)
        self.sample_ids = sample_ids
        self.seq_len = ecg_data.shape[1]
    
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        ecg = self.ecg_data[idx]
        rr = self.rr_features[idx] # [pre_rr_ratio, near_pre_rr_ratio]
        
        # DAEAC style: concat RR as channels
        pre_rr = np.full(self.seq_len, rr[0], dtype=np.float32)
        near_pre = np.full(self.seq_len, rr[1], dtype=np.float32)
        
        x = np.stack([ecg, pre_rr, near_pre], axis=0)[np.newaxis, :, :] # (1, 3, L)
        return (torch.from_numpy(x).float(), torch.from_numpy(rr).float(), self.labels[idx], self.patient_ids[idx], idx)

# =============================================================================
# [Trainer]
# =============================================================================

class ResUTrainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, exp_dir):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.exp_dir = exp_dir
        self.best = {m: {'value': -float('inf'), 'state': None} for m in ['macro_auroc', 'macro_auprc', 'macro_f1']}

    def fit(self, epochs):
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            t_loss, t_corr, t_tot = 0, 0, 0
            
            # Progress bar for training
            # pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in self.train_loader:
                ecg, rr, labels, _, _ = batch
                ecg, rr, labels = ecg.to(self.device), rr.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                logits, _ = self.model(ecg, rr)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                t_loss += loss.item() * ecg.size(0)
                t_corr += (logits.argmax(1) == labels).sum().item()
                t_tot += labels.size(0)
            
            # Validation
            val_metrics = self.evaluate(self.valid_loader)
            self.scheduler.step()
            self._update_best(val_metrics, epoch)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f" Epoch {epoch:3d} | Train Loss: {t_loss/t_tot:.4f} Acc: {t_corr/t_tot:.4f} | "
                      f"Val AUROC: {val_metrics['macro_auroc']:.4f} AUPRC: {val_metrics['macro_auprc']:.4f}")

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                ecg, rr, labels, _, _ = batch
                ecg, rr, labels = ecg.to(self.device), rr.to(self.device), labels.to(self.device)
                logits, _ = self.model(ecg, rr)
                probs = F.softmax(logits, dim=1)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return self._calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))

    def _calculate_metrics(self, y_true, y_pred, y_probs):
        n_classes = len(CLASSES)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(n_classes), zero_division=0)
        
        y_true_onehot = np.eye(n_classes)[y_true]
        auroc_list, auprc_list = [], []
        for c in range(n_classes):
            if y_true_onehot[:, c].sum() > 0:
                auroc_list.append(roc_auc_score(y_true_onehot[:, c], y_probs[:, c]))
                auprc_list.append(average_precision_score(y_true_onehot[:, c], y_probs[:, c]))
            else:
                auroc_list.append(0.0); auprc_list.append(0.0)

        return {
            'acc': (y_true == y_pred).mean(),
            'macro_f1': np.mean(f1),
            'macro_auroc': np.mean(auroc_list),
            'macro_auprc': np.mean(auprc_list),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=range(n_classes))
        }

    def _update_best(self, metrics, epoch):
        state = copy.deepcopy(self.model.state_dict())
        for metric in self.best.keys():
            if metrics[metric] > self.best[metric]['value']:
                self.best[metric] = {'value': metrics[metric], 'state': state}
                torch.save({'epoch': epoch, 'state_dict': state}, os.path.join(self.exp_dir, f"best_{metric}.pth"))

    def load_best_model(self, metric_name):
        path = os.path.join(self.exp_dir, f"best_{metric_name}.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path)['state_dict'])

# =============================================================================
# [Main Execution]
# =============================================================================

def run_single_experiment(exp_config, exp_name, data_loaders, device):
    input_mode = exp_config['input_mode']
    model_type = exp_config['model_type']
    
    print(f"\n>> Running: {exp_name} | Mode: {input_mode} | Type: {model_type}")
    set_seed(SEED)
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    loaders = data_loaders[input_mode]
    
    # Model Setup
    model_args = {
        'in_channels': BASE_MODEL_CONFIG['in_channels'],
        'out_ch': BASE_MODEL_CONFIG['out_ch'],
        'mid_ch': BASE_MODEL_CONFIG['mid_ch'],
        'num_heads': BASE_MODEL_CONFIG['num_heads'],
        'n_rr': BASE_MODEL_CONFIG['n_rr_opt1'] if input_mode == 'opt1' else BASE_MODEL_CONFIG['n_rr_opt3']
    }
    if input_mode == 'opt3':
        model_args['n_channels_opt3'] = BASE_MODEL_CONFIG['n_channels_opt3']

    model = get_resu_model(model_type=model_type, input_mode=input_mode, nOUT=len(CLASSES), **model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    trainer = ResUTrainer(model, loaders[0], loaders[1], criterion, optimizer, scheduler, device, exp_dir)
    trainer.fit(epochs=EPOCHS)

    results = {'exp_name': exp_name, 'config': exp_config, 'status': 'success', 'full_metrics': {}}
    for metric in ["macro_auprc", "macro_auroc", "macro_f1"]:
        trainer.load_best_model(metric)
        results['full_metrics'][metric] = trainer.evaluate(loaders[2]) # Test Loader
    
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== AUTOMATED RESU EXPERIMENT (REFACTORED) ===\nDevice: {device}")
    
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data (Using robust caching utils)
    print("\n[1/3] Preparing Data (Auto-Caching enabled)...")
    
    # Opt1 Data (Default Style, 7 RR features)
    d_opt1 = {split: load_or_extract_data(recs, DATA_PATH, VALID_LEADS, OUT_LEN, split, CACHE_DIR, "default")
              for split, recs in zip(["Train", "Valid", "Test"], [DS1_TRAIN, DS1_VALID, DS2_TEST])}
    
    # Opt3 Data (DAEAC Style, 2 RR features)
    d_opt3 = {split: load_or_extract_data(recs, DATA_PATH, VALID_LEADS, OUT_LEN, split, CACHE_DIR, "daeac")
              for split, recs in zip(["Train", "Valid", "Test"], [DS1_TRAIN, DS1_VALID, DS2_TEST])}

    data_loaders = {
        'opt1': (DataLoader(ECGDataset_opt1(*d_opt1['Train']), batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
                 DataLoader(ECGDataset_opt1(*d_opt1['Valid']), batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
                 DataLoader(ECGDataset_opt1(*d_opt1['Test']), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)),
        'opt3': (DataLoader(ECGDataset_opt3(*d_opt3['Train']), batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
                 DataLoader(ECGDataset_opt3(*d_opt3['Valid']), batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
                 DataLoader(ECGDataset_opt3(*d_opt3['Test']), batch_size=BATCH_SIZE, shuffle=False, num_workers=4))
    }

    # 2. Excel Init
    print("\n[2/3] Initializing Logging...")
    excel_writer, cumulative_writer = None, None
    if os.path.exists("./model_fusion.xlsx"):
        excel_writer = ExcelResultWriter("./model_fusion.xlsx", os.path.join(OUTPUT_DIR, "ResU_results.xlsx"), classes=CLASSES)
        cumulative_writer = CumulativeExcelWriter("./model_fusion.xlsx", "./cumulative_results_ResU.xlsx", classes=CLASSES)

    # 3. Run Experiments
    print(f"\n[3/3] Running {len(EXPERIMENT_GRID)} experiments...")
    for i, exp_config in enumerate(EXPERIMENT_GRID, 1):
        try:
            res = run_single_experiment(exp_config, exp_config['name'], data_loaders, device)
            
            # Excel Logging
            for m_key, metrics in res['full_metrics'].items():
                short = m_key.replace("macro_", "")
                if excel_writer: 
                    excel_writer.write_metrics(exp_config['name'], metrics, short)
                    if 'confusion_matrix' in metrics: excel_writer.write_confusion_matrix(exp_config['name'], metrics['confusion_matrix'], short)
                if cumulative_writer: cumulative_writer.append_result(exp_config['name'], metrics, exp_config, short)
            
            print(f"  -> Saved results for {exp_config['name']}")
        except Exception as e:
            print(f"  -> ERROR in {exp_config['name']}: {e}")
            import traceback; traceback.print_exc()

    print("\n=== ALL COMPLETED ===")

if __name__ == '__main__':
    main()