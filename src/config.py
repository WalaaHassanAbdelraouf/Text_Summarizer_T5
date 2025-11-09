import os

# ============================================================================
# PATHS
# ============================================================================
DATA_PATH = './data/news_summary_more.csv'
MODEL_SAVE_PATH = './models'
RESULTS_DIR = './results'
LOGS_DIR = './logs'

# ============================================================================
# DATA PARAMETERS
# ============================================================================
SAMPLE_SIZE = 7000  
TEST_SIZE = 0.2
VAL_SPLIT = 0.5
RANDOM_STATE = 42

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
MODEL_NAME = 't5-small'
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4
LOGGING_STEPS = 100

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================
USE_FP16 = True
USE_GRADIENT_CHECKPOINTING = True
OPTIMIZER = "adamw_torch"
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
NUM_BEAMS = 4
LENGTH_PENALTY = 2.0
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 3

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["WANDB_DISABLED"] = "true"