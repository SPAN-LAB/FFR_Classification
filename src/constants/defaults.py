NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.1
MIN_DELTA = 0.001
PATIENCE = 20
LR_PATIENCE = PATIENCE // 2
VALIDATION_RATIO = 0.10

SUBAVERAGE_SIZE  = 5
NUM_FOLDS        = 5
TRIM_START_TIME  = 50  # in milliseconds
TRIM_END_TIME    = 250 # in milliseconds
TRAINING_OPTIONS = {
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": LEARNING_RATE,
    "min_delta": MIN_DELTA,
    "patience": PATIENCE,
    "lr_patience": LR_PATIENCE,
    "validation_ratio": VALIDATION_RATIO
}