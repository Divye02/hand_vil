from visual_im.settings import *
import pickle
from mjrl_mod.samplers.base_sampler import do_rollout
import mjrl.envs
from mjrl_mod.utils.gym_env import GymEnv
import numpy as np

def make_train_data(env, policy):
    num_batches = 5 # number of files as 1 batch = 1 file
    SMALL_TRAIN_TRAJS = TRAIN_TRAJS // 4
    batch_size = SMALL_TRAIN_TRAJS // num_batches
    for i in range(num_batches):
        paths = np.array(do_rollout(N=batch_size, policy=policy, env=env, use_mean=True))
        train_file = os.path.join(TRAIN_DATA_DIR, 'train_paths' + '_batch' + str(i) + '_horizon150.pickle')
        pickle.dump(paths, open(ensure_dir(train_file), 'wb'))


def make_val_data(env, policy):
    num_batches = 1
    batch_size = TEST_TRAJS // num_batches
    for i in range(num_batches):
        paths = np.array(do_rollout(N=batch_size, policy=policy, env=env))
        val_file = os.path.join(VAL_DATA_DIR, 'val_paths' + '_batch' + str(i) + '_horizon150.pickle')
        pickle.dump(paths, open(ensure_dir(val_file), 'wb'))


def main():
    # env = GymEnv('mjrl_reacher7dof-v0')
    env = GymEnv('mjrl_ant-v0')
    expert_policy = pickle.load(open(EXPERT_POLICY_PATH, 'rb'))
    
    make_val_data(env, expert_policy)
    

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

if __name__ == '__main__':
    main()
    #TODO: add argparse for ease of use w/ train vs val + batch size, etc.
