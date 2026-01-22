import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# Config에서 자동 계산된 변수들 Import
import config
from utils import set_seed, load_or_extract_data
from model import MACNN_SE
from dataloader import ECGDataset, DAEACDataset
from train import train_one_epoch, validate, save_model
from logger import TrainingLogger, print_epoch_header, print_epoch_stats

# 

# 1. 초기화
set_seed(config.SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
exp_dir = config.create_experiment_dir()

print(f"\n>>> MODE: {'DAEAC (Paper)' if config.is_daeac_style() else 'Standard Experiment'}")
print(f">>> Style: {config.CURR_STYLE}, Input Length: {config.CURR_OUT_LEN}, Lead: {config.CURR_LEAD}")

# 2. 데이터 로드
print("\n[1/3] Loading Data...")

def get_loader(records, split_name, shuffle=False):
    data = load_or_extract_data(
        record_list=records, base_path=config.DATA_PATH, valid_leads=config.VALID_LEADS,
        out_len=config.CURR_OUT_LEN, split_name=split_name, extraction_style=config.CURR_STYLE
    )
    # config.is_daeac_style() 결과에 따라 Dataset 클래스 자동 선택
    DS_Class = DAEACDataset if config.is_daeac_style() else ECGDataset
    return DataLoader(DS_Class(*data), batch_size=config.BATCH_SIZE, shuffle=shuffle, num_workers=4, pin_memory=True)

train_loader = get_loader(config.DS1_TRAIN, "Train", shuffle=True)
valid_loader = get_loader(config.DS1_VALID, "Valid")
test_loader  = get_loader(config.DS2_TEST,  "Test")

# 3. 모델 생성 (config 딕셔너리 언패킹)
print("\n[2/3] Building MACNN_SE Model...")
model = MACNN_SE(
    rr_dim=config.RR_FEATURE_DIMS[config.RR_FEATURE_OPTION],
    **config.MACNN_SE_CONFIG
).to(device)

# 4. 학습 준비
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.99 ** (step / 200))

# Class Weights (명세대로 imbalance 대응)
labels = train_loader.dataset.labels
cnt = np.bincount(labels, minlength=4)
weights = torch.tensor(1.0 / (cnt + 1e-8), dtype=torch.float32).to(device)
weights = weights / weights.sum() * 4

logger = TrainingLogger(os.path.join(exp_dir, 'runs'))
best_f1 = 0.0

# 5. Training Loop
print("\n[3/3] Start Training...")
print_epoch_header()

for epoch in range(1, config.EPOCHS + 1):
    t0 = time.time()
    
    t_loss, t_met, *_ = train_one_epoch(model, train_loader, epoch, optimizer, device, scheduler, weights)
    v_loss, v_met, *_ = validate(model, valid_loader, device)
    
    # 로깅 및 상태 출력
    print_epoch_stats(epoch, t_loss, t_met['acc'], optimizer.param_groups[0]['lr'], 'Train')
    print_epoch_stats(epoch, v_loss, v_met['acc'], optimizer.param_groups[0]['lr'], 'Valid')
    logger.log_epoch(epoch, t_loss, t_met, 'train')
    logger.log_epoch(epoch, v_loss, v_met, 'valid')

    # Best Model 저장 (F1 기준)
    if v_met['macro_f1'] > best_f1:
        best_f1 = v_met['macro_f1']
        save_model(model, optimizer, epoch, v_met, os.path.join(exp_dir, 'best_weights', f'best_model_{config.EXP_NAME}.pth'))
        print(f"  [New Best F1]: {best_f1:.4f}")

    print(f"  Duration: {time.time()-t0:.1f}s | " + "-"*50)

logger.close()