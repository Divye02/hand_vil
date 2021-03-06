# Clip and BinAction stuff that we had played and tested with, no logner using BinAction.
# ToCudaTensor to use GPU
from data_gen.transforms import ClipAction, ToCudaTensor
# loading dataset of observation, action, and image of state/sequence of
# images of previous states and current state [for context to capture motion info]
from data_gen.dataset import get_dataset_from_files
# Torch dataloader for loading training and validation data into network [data as described above].
from torch.utils.data import DataLoader
# Local settings for making Divye happy [TODO: You may want to fill this, guruji]
from settings import *
# Multi-layer perceptron policy
from mjrl_mod.policies.gaussian_cnn import CNN
# DAgger [Dataset Aggregation] algorithm, as described in Ross et. al. paper
from mjrl_mod.algos.dagger import Dagger
from mjrl_mod.utils.gym_env import GymEnv
from torchvision import transforms
from mjrl_mod.samplers.base_sampler import do_rollout
from gym.envs.registration import register

from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl_mod.utils.train_agent import train_agent

import numpy as np
import pickle
import glob
import time as timer


def train_expert_policy(config):
    print('-' * 80)
    previous_dir = os.getcwd()
    ensure_dir(GEN_DATA_DIR)
    os.chdir(GEN_DATA_DIR)

    print('Training Expert')
    e = make_gym_env(config['env_id'], config)
    policy = MLP(e.spec, hidden_sizes=(32, 32), seed=config['seed'])
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
    agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=config['seed'], save_logs=True)

    job_name = '%s_expert' % config['env_name']
    # Need to change where it dumps the policy
    train_agent(job_name=job_name,
                agent=agent,
                seed=config['seed'],
                niter=30,
                gamma=0.995,
                gae_lambda=0.97,
                num_cpu=1,
                sample_mode='trajectories',
                num_traj=200,
                save_freq=5,
                evaluation_rollouts=5)
    os.chdir(previous_dir)
    os.rename(os.path.join(GEN_DATA_DIR, job_name, 'iterations/best_policy.pickle'),
              os.path.join(EXPERT_POLICIES_DIR, EXPERT_POLICIES[config['env_name']]))
    print('-' * 80)

def gen_data_from_expert(config):
    print('-' * 80)

    train_dir = os.path.join(config['main_dir'], 'train_data')
    val_dir = os.path.join(config['main_dir'], 'val_data')

    e = make_gym_env(config['env_id'], config)
    try:
        gen_data(e, train_dir, config['num_files_train'], config['train_traj_per_file'], config)
    except Exception as e:
        os.rmdir(train_dir)
        raise
    try:
        gen_data(e, val_dir, config['num_files_val'], config['val_traj_per_file'], config)
    except Exception as e:
        os.rmdir(val_dir)
        raise

    del (e)


def do_dagger(config):
    config['viz_policy_folder_dagger'] = 'dagger_%s_viz_policy' % config['env_name']
    viz_policy_folder_dagger = os.path.join(config['main_dir'], config['viz_policy_folder_dagger'])

    print('-' * 80)
    if os.path.exists(viz_policy_folder_dagger):
        print('DAgger: Viz policy already exists')
        return
    print('DAgger: Training viz policy now')

    ensure_dir(viz_policy_folder_dagger)
    train_dataloader, val_dataloader, transformed_train_dataset, transformed_val_dataset = get_dataloaders_datasets(config)

    # policy = MLP(e.spec, hidden_sizes=(64,64), seed=SEED)
    e = make_gym_env(config['env_id'], config)
    robot_info_dim = None
    if config['has_robot_info']:
        robot_info_dim = e.env.env.robot_info_dim
    policy = CNN(action_dim=transformed_train_dataset.action_dim,
                 use_seq=True,
                 robot_info_dim=robot_info_dim,
                 action_stats=transformed_train_dataset.get_action_stats(),
                 robot_info_stats=transformed_train_dataset.get_robot_info_stats(),
                 use_late_fusion=config['use_late_fusion'], use_cuda=config['use_cuda'])

    ts = timer.time()

    expert_policy = pickle.load(open(get_expert_policy_path(config['env_name'], config), 'rb'))
    dagger_algo = Dagger(
        dagger_epochs=config['dagger_epoch'],
        expert_policy=expert_policy,
        viz_policy=policy,
        old_data_loader=train_dataloader,
        val_data_loader=val_dataloader,
        log_dir=os.path.join(config['id'], 'dagger'),
        pol_dir_name=viz_policy_folder_dagger,
        save_epoch=1,
        beta_decay=config['beta_decay'],
        beta_start=config['beta_start'],
        env=e,
        lr=config['lr'],
        num_traj_gen=config['gen_traj_dagger_ep'],
        camera_name=config['camera_name'],
        has_robot_info=config['has_robot_info'],
        seed=config['seed'] + (config['num_files_train'] * config['train_traj_per_file']),
        trainer_epochs=config['trainer_epochs'],
        eval_num_traj=config['eval_num_traj'],
        sliding_window=config['sliding_window'],
        device_id=config['device_id'],
        use_cuda=config['use_cuda'])

    dagger_algo.train()
    trained_policy = dagger_algo.viz_policy

    print("time taken = %f" % (timer.time() - ts))
    del (e)


def get_dataloaders_datasets(config):
    train_dir = os.path.join(config['main_dir'], 'train_data')
    val_dir = os.path.join(config['main_dir'], 'val_data')

    train_path_files = glob.glob(os.path.join(train_dir, '*'))
    val_path_files = glob.glob(os.path.join(val_dir, '*'))

    if config['use_cuda']:
        transforms_list = [ClipAction(), ToCudaTensor()]
    else:
        transforms_list = [ClipAction()]

    transformed_train_dataset = get_dataset_from_files(train_path_files,
                            transform=transforms.Compose(transforms_list))
    transformed_val_dataset = get_dataset_from_files(val_path_files,
                            transform=transforms.Compose(transforms_list))

    train_dataloader = DataLoader(transformed_train_dataset,
                                  batch_size=config['batch_size_viz_pol'],
                                  shuffle=True,
                                  num_workers=4)
    val_dataloader = DataLoader(transformed_val_dataset,
                                batch_size=config['batch_size_viz_pol'],
                                shuffle=True,
                                num_workers=4)
    return train_dataloader, val_dataloader, transformed_train_dataset, transformed_val_dataset


def gen_data(env, data_dir, num_files, trajs_per_file, config):
    if os.path.exists(data_dir):
        print('%s folder already exists' % os.path.basename(data_dir))
        return

    ensure_dir(data_dir)
    print('Generating %s' % os.path.basename(data_dir))
    expert_policy_path = get_expert_policy_path(config['env_name'], config)
    expert_policy = pickle.load(open(expert_policy_path, 'rb'))

    for i in range(num_files):
        seed = config['seed'] + i * trajs_per_file
        paths = np.array(
            do_rollout(N=trajs_per_file, policy=expert_policy, env=env, pegasus_seed=seed, use_mean=True, save_img=True,
                       camera_name=config['camera_name'], device_id=config['device_id']))
        train_file = os.path.join(data_dir, 'train_paths_%s_batch_%d.pickle' % (config['env_name'], i))
        pickle.dump(paths, open(train_file, 'wb'))


def make_gym_env(id, config):
    e = GymEnv(id, use_tactile=config['use_tactile'])
    config['has_robot_info'] = e.has_robot_info_attr()
    config['env_spec'] = e.spec.as_dict()
    return e


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_expert_policy_path(env_name, config):
    exp_p_p = os.path.join(EXPERT_POLICIES_DIR, EXPERT_POLICIES[env_name])
    print('Using: %s' % exp_p_p)
    return exp_p_p


def register_env(config):
    register(
        id=config['env_id'],
        entry_point=ENTRY_POINT[config['env_name']],
        max_episode_steps=config['horizon_il'],
    )

def viz_pol(pol_path, config):
    print('Visualizing %s' % pol_path)
    # id = 'mjrl_half_cheetah-v0'
    e = GymEnv(config['env_id'], use_tactile=config['use_tactile'])
    p_p = os.path.join(POLICIES_DIR, pol_path)
    # p_p = EXPERT_POLICY_PATH
    policy = pickle.load(open(p_p, 'rb'))
    print('usind %d horizon', e.horizon)
    policy.model.eval()
    policy.old_model.eval()
    # reward, _, _ = e.evaluate_policy(policy,
    #                     num_episodes=10, mean_action=True, use_seq=True,
    #                     camera_name=CAMERA_NAME[env_name], seed=500)

    # print('reward', reward)

    # e.visualize_policy(policy, num_episodes=20, horizon=100, mode='evaluation', use_img=False, use_seq=False, bin=args.bin, frame_size=FRAME_SIZE, camera_name=CAMERA_NAME[env_name])
    e.visualize_policy_offscreen(ensure_dir(VIDOES_FOLDER), env_name, policy=policy, num_episodes=5, horizon=e.horizon, mode='evaluation', use_img=True, use_seq=True, camera_name=CAMERA_NAME[env_name], pickle_dump=False)
    del(e)

