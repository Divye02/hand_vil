from local_settings import *
import os

GEN_DATA_DIR = os.path.join(MAIN_DIR, 'gen_data')
'''
DEFAULT_CONFIG = dict(
    # Other config stuff
    device_id = 0,
    env_name='point_mass',
    seed=500,
    camera_name='top_view',
    use_cuda=False,
    id_post='test20',
    train_expert=False,


    # used got training expert policy
    delta = 0.01,
    traj_budget_expert = 12500,
    num_traj_expert= 50,

    # used for gen data
    train_traj_per_file= 1,
    #train_traj_per_file= 1,
    num_files_train= 1,
    val_traj_per_file = 1,
    #val_traj_per_file = 1,
    num_files_val= 1,

    batch_size_viz_pol = 128,
    lr=0.0003,

    # used for DAgger
    dagger_epoch=1,
    #dagger_epoch=1,
    beta_decay=0.9,
    beta_start=1.0,
    # set the window to 0 for aggregation
    sliding_window=1,
    gen_traj_dagger_ep=1,
    eval_num_traj=1,
    trainer_epochs=1,
    horizon_il=50,
    use_tactile=True,
    use_late_fusion=True,
)
'''
# DEFAULT_CONFIG = dict(
#     # Other config stuff
#     device_id = 0,
#     env_name='point_mass',
#     seed=500,
#     camera_name='top_view',
#     use_cuda=True,
#     id_post='test22',
#     train_expert=False,
#
#
#     # used got training expert policy
#     delta = 0.01,
#     traj_budget_expert = 12500,
#     num_traj_expert= 50,
#
#     # used for gen data
#     train_traj_per_file=50,
#     #train_traj_per_file= 1,
#     num_files_train= 1,
#     val_traj_per_file = 50,
#     #val_traj_per_file = 1,
#     num_files_val= 1,
#
#     batch_size_viz_pol = 128,
#     lr=0.0003,
#
#     # used for DAgger
#     dagger_epoch=20,
#     #dagger_epoch=1,
#     beta_decay=0.9,
#     beta_start=1.0,
#     # set the window to 0 for aggregation
#     sliding_window=100,
#     gen_traj_dagger_ep=100,
#     eval_num_traj=50,
#     trainer_epochs=10,
#     horizon_il=50,
#     use_tactile=True,
#     use_late_fusion=True,
# )

DEFAULT_CONFIG = dict(
     id_post='testH2_public',
     beta_decay= 0.2,
     beta_start= 1.0,
     num_traj_expert= 50,
     dagger_epoch= 20,
     horizon_il= 150,
     has_robot_info= True,
     train_traj_per_file= 50,
     gen_traj_dagger_ep= 50,
     env_name= "hand_hammer",
     delta= 0.01,
     sliding_window= 100,
     seed= 1000,
     expert_policy_folder= "hand_hammer_expert",
     traj_budget_expert= 12500,
     viz_policy_folder_dagger= "dagger_hand_hammer_viz_policy",
     id= "hand_hammer_id_robotinfo_seed1000_beta0.2_ep20",
     lr= 0.0003,
     use_tactile= False,
     trainer_epochs= 5,
     eval_num_traj= 100,
     num_files_train= 1,
     batch_size_viz_pol= 128,
     val_traj_per_file= 5,
     num_files_val= 1,
     bc_epoch= 20,
     use_cuda=True,
     camera_name='vil_camera',
     train_expert=False,
     device_id=0,
     use_late_fusion=True,
 )

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

EXPERT_POLICIES = {
    # Add  Other envs here
    'hand_pickup': 'relocate.pickle',
    'hand_hammer': 'hammer.pickle',
    'hand_pen': 'pen.pickle',
    'hand_door': 'door.pickle',
    'point_mass': 'point_mass.pickle',
}

ENTRY_POINT = {
    'hand_pickup': 'mjrl_mod.envs.hand_hammer:HandPickup',
    'hand_hammer': 'mjrl_mod.envs.hand_hammer:HandHammer',
    'hand_pen': 'mjrl_mod.envs.hand_hammer:HandPen',
    'hand_door': 'mjrl_mod.envs.hand_hammer:HandDoor',
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
