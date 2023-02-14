import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_VIS = "data/vis/train"
VAL_DIR_VIS = "data/vis/val"
TRAIN_DIR_IR = "data/ir/train"
VAL_DIR_IR = "data/ir/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
# LAMBDA_GP = 10
NUM_EPOCHS = 2
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"



