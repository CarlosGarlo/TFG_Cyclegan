import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "data2/train"
VAL_DIR = "data2/val"
TEST_DIR = "data/test"

MODEL_DIR_SN = "model_saved/SunnyNight"
MODEL_DIR_SC = "model_saved/SunnyCloudy"
MODEL_DIR_NC = "model_saved/NightCloudy"

BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 100

LOAD_MODEL = False
SAVE_MODEL = True

# CHECKPOINT_GEN_H = "genh.pth.tar"
# CHECKPOINT_GEN_Z = "genz.pth.tar"
# CHECKPOINT_CRITIC_H = "critich.pth.tar"
# CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

CHECKPOINT_GEN_S_SN = config.MODEL_DIR_SN + "/genS_sn.pth.tar"
CHECKPOINT_GEN_N_SN = config.MODEL_DIR_SN + "/genN_sn.pth.tar"
CHECKPOINT_DISCRIM_S_SN = config.MODEL_DIR_SN + "/discrimS_sn.pth.tar"
CHECKPOINT_DISCRIM_N_SN = config.MODEL_DIR_SN + "/discrimN_sn.pth.tar"

CHECKPOINT_GEN_S_SC = config.MODEL_DIR_SC + "/genS_sc.pth.tar"
CHECKPOINT_GEN_C_SC = config.MODEL_DIR_SC + "/genC_sc.pth.tar"
CHECKPOINT_DISCRIM_S_SC = config.MODEL_DIR_SC + "/discrimS_sc.pth.tar"
CHECKPOINT_DISCRIM_C_SC = config.MODEL_DIR_SC + "/discrimC_sc.pth.tar"

CHECKPOINT_GEN_N_NC = config.MODEL_DIR_NC + "/genN_nc.pth.tar"
CHECKPOINT_GEN_C_NC = config.MODEL_DIR_NC + "/genC_nc.pth.tar"
CHECKPOINT_DISCRIM_N_NC = config.MODEL_DIR_NC + "/discrimN_nc.pth.tar"
CHECKPOINT_DISCRIM_C_NC = config.MODEL_DIR_NC + "/discrimC_nc.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=512, height=128),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
