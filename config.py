"""
ECG Classification Configuration

모든 실험 설정을 전역 변수로 관리합니다.
연구자는 이 파일만 수정하면 실험 환경을 변경할 수 있습니다.

=============================================================================
수정 가이드:
- 모델 변경: MODEL_NAME과 MODEL_CONFIG 수정
- Loss 변경: LOSS_NAME과 LOSS_CONFIG 수정
- Dataset 변경: DATASET_NAME 수정
- Optimizer 변경: OPTIMIZER_NAME과 OPTIMIZER_CONFIG 수정
- Scheduler 변경: SCHEDULER_NAME과 SCHEDULER_CONFIG 수정
=============================================================================
"""

import os
from datetime import datetime

# =============================================================================
# 1. 실험 기본 설정
# =============================================================================
EXP_NAME = "DAEAC_baseline"
SEED = 1234
DEVICE = "cuda"  # "auto", "cuda", "cpu", "cuda:0", etc.

# =============================================================================
# 2. 경로 설정
# =============================================================================
DATA_PATH = '/home/work/Ryuha/ECG_CrossAttention/data/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_PATH = './results/'
CACHE_PATH = './dataset/'

# =============================================================================
# 3. 데이터 설정
# =============================================================================
DATASET_NAME = "daeac"  # Registry: "ecg_standard", "daeac"

# 데이터 처리 옵션
RR_FEATURE_OPTION = "opt2"  # "opt1" (7 features), "opt2" (2 features for DAEAC)
SIGNAL_LENGTH = 128         # ECG 신호 길이
EXTRACTION_STYLE = "daeac"  # "default", "daeac"

# RR feature 차원 (자동 설정)
RR_FEATURE_DIMS = {"opt1": 7, "opt2": 2}
RR_DIM = RR_FEATURE_DIMS[RR_FEATURE_OPTION]

# =============================================================================
# 입력 형태 설정
# =============================================================================
# opt2 (DAEAC): 입력 (N, 1, 3, 128) - ECG + pre_RR + near_pre_RR (early fusion)
# opt1 (Standard): 입력 (N, 1, 1, 128) - ECG만, RR은 후단 fusion (late fusion)
LEAD = 3 if RR_FEATURE_OPTION == "opt2" else 1

# =============================================================================
# 4. 클래스 정의 (AAMI 표준)
# =============================================================================
CLASSES = ['N', 'S', 'V', 'F']
NUM_CLASSES = len(CLASSES)
LABEL_TO_ID = {'N': 0, 'S': 1, 'V': 2, 'F': 3}
ID_TO_LABEL = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}

# 개별 비트 기호 → AAMI 클래스 매핑
LABEL_GROUP_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
}

# =============================================================================
# 5. 데이터 분할 (Chazal Protocol)
# =============================================================================
VALID_LEADS = ['MLII', 'V1', 'V2', 'V4', 'V5']

DS1_TRAIN = [
    '101', '106', '108', '109', '112', '115', '116', '118', '119',
    '122', '201', '203', '209', '215', '223', '230', '208'
]
DS1_VALID = ['114', '124', '205', '207', '220']
DS2_TEST = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200',
    '202', '210', '212', '213', '214', '219', '221', '222', '228',
    '231', '232', '233', '234'
]

# =============================================================================
# 6. 모델 설정
# =============================================================================
MODEL_NAME = "macnn_se"  # Registry: "macnn_se", "macnn", "acnn", etc.

MODEL_CONFIG = {
    # 기본 구조
    # 입력: (N, 1, lead, length) - in_channels=1, lead=LEAD
    "in_channels": 1,
    "lead": LEAD,  # 1 (ECG only) or 3 (ECG + 2 RR ratios)
    "num_classes": NUM_CLASSES,
    "reduction": 16,
    "dilations": (1, 6, 12, 18),
    "dropout": 0.0,

    # Activation
    "act_func": "tanh",
    "final_act_func": "tanh",

    # ASPP
    "aspp_bn": True,
    "aspp_act": True,

    # Late Fusion (opt1 전용, lead=1일 때만 사용)
    # opt2 (DAEAC)는 입력에 RR이 이미 포함 (early fusion) → fusion_type=None
    # opt1일 때 ablation: None, "concat", "concat_proj", "mhca"
    "fusion_type": None,
    "fusion_emb": 64,
    "fusion_expansion": 2,
    "fusion_num_heads": 1,
    "rr_dim": RR_DIM,

    # Residual classifier
    "apply_residual": False,
}

# =============================================================================
# 7. Loss 설정
# =============================================================================
LOSS_NAME = "cross_entropy"  # Registry: "cross_entropy", "focal", "label_smoothing"

LOSS_CONFIG = {
    "use_class_weights": True,   # 클래스 불균형 가중치 사용
    "label_smoothing": 0.0,      # Label smoothing (0.0 = 사용 안함)
}

# Focal Loss 전용 설정 (LOSS_NAME = "focal" 일 때)
FOCAL_LOSS_CONFIG = {
    "gamma": 2.0,
    "alpha": None,  # None이면 class weights 사용
}

# =============================================================================
# 8. Optimizer 설정
# =============================================================================
OPTIMIZER_NAME = "adam"  # Registry: "adam", "adamw", "sgd"

OPTIMIZER_CONFIG = {
    "lr": 0.005,
    "weight_decay": 1e-4,

    # Adam/AdamW specific
    "betas": (0.9, 0.999),
    "eps": 1e-8,

    # SGD specific (OPTIMIZER_NAME = "sgd" 일 때)
    "momentum": 0.9,
    "nesterov": False,
}

# =============================================================================
# 9. Scheduler 설정
# =============================================================================
SCHEDULER_NAME = "lambda_decay"  # Registry: "lambda_decay", "step", "cosine", "none"

SCHEDULER_CONFIG = {
    # Lambda decay (DAEAC style): lr = lr * decay_rate^(step/decay_steps)
    "decay_rate": 0.99,
    "decay_steps": 200,

    # Step LR
    "step_size": 30,
    "gamma": 0.1,

    # Cosine annealing
    "t_max": 100,
    "eta_min": 1e-6,

    # Warmup
    "warmup_epochs": 0,
    "warmup_lr": 1e-6,
}

# =============================================================================
# 10. 학습 설정
# =============================================================================
EPOCHS = 5
BATCH_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = True

# Gradient clipping (None이면 사용 안함)
GRADIENT_CLIP_VAL = None

# Early stopping
EARLY_STOPPING = False
PATIENCE = 20
MONITOR_METRIC = "macro_auprc"  # 모니터링할 메트릭
MONITOR_MODE = "max"            # "max" or "min"

# Best model 저장 기준
SAVE_BEST_METRICS = ["macro_auprc", "macro_auroc", "macro_recall"]

# 체크포인트 저장 주기
SAVE_EVERY = 10  # N epoch마다 저장 (0이면 저장 안함)

# =============================================================================
# 11. Helper Functions
# =============================================================================
def get_device():
    """디바이스 자동 설정"""
    import torch
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def create_experiment_dir():
    """실험 결과 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_PATH, f'{timestamp}_{EXP_NAME}')
    os.makedirs(exp_dir, exist_ok=True)

    # 서브 디렉토리
    for subdir in ['checkpoints', 'best_models', 'logs', 'results']:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    return exp_dir


def is_daeac_style() -> bool:
    """DAEAC 스타일 여부"""
    return RR_FEATURE_OPTION == "opt2"


def print_config():
    """현재 설정 출력"""
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"Experiment : {EXP_NAME}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Loss       : {LOSS_NAME}")
    print(f"Optimizer  : {OPTIMIZER_NAME}")
    print(f"Scheduler  : {SCHEDULER_NAME}")
    print(f"Dataset    : {DATASET_NAME}")
    print("-" * 60)
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch Size : {BATCH_SIZE}")
    print(f"LR         : {OPTIMIZER_CONFIG['lr']}")
    print(f"Device     : {DEVICE}")
    print("=" * 60 + "\n")
