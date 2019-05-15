import pickle
from mjrl_mod.utils.gym_env import GymEnv
from settings import *
import mjrl_mod.envs

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


CAMERA_NAME = 'vil_camera'
ENV_NAME = 'hand_hammer'
VIZ_FOLDER = 'hand_hammer_videos'
FULL_POLICY_PATH = ''


def main():
    e = GymEnv(VIZ_ENV_IDS[ENV_NAME], use_tactile=True)
    policy = pickle.load(open(FULL_POLICY_PATH, 'rb'))
    print('usind %d horizon', e.horizon)
    policy.model.eval()
    policy.old_model.eval()
    e.visualize_policy_offscreen(ensure_dir(os.path.join(VIDOES_FOLDER, VIZ_FOLDER)) + '/', ENV_NAME, policy=policy,
                                 num_episodes=3, horizon=e.horizon, mode='evaluation', use_img=True, use_seq=True,
                                 camera_name=CAMERA_NAME[ENV_NAME], pickle_dump=False)
    del (e)


if __name__ == '_main_':
    main()