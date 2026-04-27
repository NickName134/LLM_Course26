# config.py

import os
from datetime import datetime

# -----model-------------------------------------------------------------
MODEL = "distilbert-base-uncased"
NUM_LABELS = 2

# -----dataset----------------------------------------------------------
DATASET = "glue"
DATASET_CONFIG = "sst2"   
MAX_LENGTH = 32

# -----training---------------------------------------------------------
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
SAVE_STRATEGY = "epoch"  # o "steps"

# -- output dir con parametri embedded --
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = os.path.join(
    "checkpoints",
    f"{MODEL}_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}_ep{EPOCHS}_{timestamp}"
)


# -----evaluation-------------------------------------------------------
EVAL_STRATEGY = "epoch"  # o "steps"
EVAL_STEPS = 100         # se EVAL_STRATEGY="steps", ogni quanti step valutare
LOGGING_STEPS = 10       # ogni quanti step loggare metriche (loss, acc, ecc.)

# -----reproducibility------------------------------------------------------
SEED = 42
