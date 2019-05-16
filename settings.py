from local_settings import *
import os
from config_main import DEFAULT_CONFIG
GEN_DATA_DIR = os.path.join(MAIN_DIR, 'gen_data')

RES_DIR = os.path.join(GEN_DATA_DIR, 'results')
VIDOES_FOLDER = os.path.join(RES_DIR, 'videos')
PLOTS_FOLDER = os.path.join(RES_DIR, 'plots')


DATA_DIR = os.path.join(GEN_DATA_DIR, 'data')
LOG_DIR = os.path.join(DATA_DIR, 'logs')

POLICIES_DIR = os.path.join(DATA_DIR, 'policies')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train_data')
VAL_DATA_DIR = os.path.join(DATA_DIR, 'val_data')

EXPERT_POLICIES_DIR = os.path.join(MAIN_DIR, 'expert_policies')
TRAIN_TRAJS = 5000
TEST_TRAJS = 100

ENV_ID = {
    'hand_pickup': 'mjrl_SHAP_slide_pickup-v42',
    'hand_hammer': 'mjrl_hammer-v0',
    'hand_pen': 'mjrl_pen_reposition-v2',
    'hand_door': 'mjrl_SHAP_door_handle-v5',
    'point_mass': 'mjrl_point_mass-v1',
}

VIZ_ENV_IDS = {
    'hand_hammer': 'mjrl_hand_hammer_viz-v0',
    'hand_door': 'mjrl_hand_door_viz-v0',
    'hand pickup': 'mjrl_hand_pickup_viz-v0',
    'hand_pen': 'mjrl_hand_pen_viz-v0'
}

EXPERT_POLICIES = {
    # Add  Other envs here
    'hand_pickup': 'relocate.pickle',
    'hand_hammer': 'hammer.pickle',
    'hand_pen': 'pen.pickle',
    'hand_door': 'door.pickle',
    'point_mass': 'point_mass.pickle',
}

ENTRY_POINT = {
    'hand_pickup': 'mjrl_mod.envs.hand_pickup:HandPickup',
    'hand_hammer': 'mjrl_mod.envs.hand_hammer:HandHammer',
    'hand_pen': 'mjrl_mod.envs.hand_pen:HandPen',
    'hand_door': 'mjrl_mod.envs.hand_door:HandDoor',
    'point_mass': 'mjrl_mod.envs:PointMassEnv',
}


FRAME_SIZE = (128, 128)

CAMERA_NAME = {
    'hand_pickup': 'vil_camera',
    'hand_hammer': 'vil_camera',
    'hand_pen': 'vil_camera',
    'hand_door': 'vil_camera',
    'point_mass': 'top_view',
}
