import os
from datetime import datetime

# =============================================================================
# 1. 메인 컨트롤 타워
# =============================================================================
OPTION_PAPER = 2  # 1: 일반 실험, 2: DAEAC 논문 재현 모드
EXP_NAME = "DAEAC_baseline"

# OPTION_PAPER에 따른 자동 옵션 분기
if OPTION_PAPER == 2:
    RR_FEATURE_OPTION = "opt2"  # DAEAC paper (2 features)
else:
    RR_FEATURE_OPTION = "opt1"  # 기존 실험용 (7 features)

# =============================================================================
# 2. Hyperparameters & Paths
# =============================================================================
BATCH_SIZE = 256
EPOCHS = 300
LR = 0.005
WEIGHT_DECAY = 1e-4
SEED = 1234

DATA_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_PATH = './ECG_Results/'

# =============================================================================
# 3. 옵션별 세부 맵핑
# =============================================================================
RR_FEATURE_DIMS = {"opt1": 7, "opt2": 2}
RR_OPTION_TO_STYLE = {"opt1": "default", "opt2": "daeac"}
STYLE_OUT_LEN = {"default": 720, "daeac": 128}
RR_OPTION_TO_LEAD = {"opt1": 1, "opt2": 3}

CURR_STYLE = RR_OPTION_TO_STYLE[RR_FEATURE_OPTION]
CURR_OUT_LEN = STYLE_OUT_LEN[CURR_STYLE]
CURR_LEAD = RR_OPTION_TO_LEAD[RR_FEATURE_OPTION]

# =============================================================================
# 4. Classes & Label Mappings (AAMI 표준) - utils.py에서 필요함
# =============================================================================
CLASSES = ['N', 'S', 'V', 'F']
LABEL_TO_ID = {'N': 0, 'S': 1, 'V': 2, 'F': 3}
ID_TO_LABEL = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}

# 개별 비트 기호를 AAMI 4개 클래스로 묶어주는 맵
LABEL_GROUP_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
}

# =============================================================================
# 5. Model Architecture (MACNN_SE)
# =============================================================================
MACNN_SE_CONFIG = {
    'reduction': 16,
    'aspp_bn': True,
    'aspp_act': True,
    'lead': CURR_LEAD,
    'p': 0.0,
    'dilations': (1, 6, 12, 18),
    'act_func': 'tanh',
    'f_act_func': 'tanh',
    'apply_residual': False,
    'fusion_type': 'none',
    'fusion_emb': 64,
    'fusion_expansion': 2,
    'num_heads': 1,
}

# =============================================================================
# 6. Data Split (Chazal)
# =============================================================================
VALID_LEADS = ['MLII', 'V1', 'V2', 'V4', 'V5']
DS1_TRAIN = ['101', '106', '108', '109', '112', '115', '116', '118', '119', '122', '201', '203', '209', '215', '223', '230', '208']
DS1_VALID = ['114', '124', '205', '207', '220']
DS2_TEST = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

# =============================================================================
# 7. Public Helper Functions
# =============================================================================
def is_daeac_style() -> bool:
    return RR_FEATURE_OPTION == "opt2"

def create_experiment_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_PATH, f'experiment_{timestamp}_{EXP_NAME}')
    os.makedirs(exp_dir, exist_ok=True)
    for sub in ['correct', 'incorrect', 'best_weights']:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    return exp_dir