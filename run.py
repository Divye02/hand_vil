from run_utils import *
# Self-explanatory python stuff.
import time as timer
import json
import argparse
import os


def main(config):
    dump(config)

    ts = timer.time()
    register_env(config)

    if config['train_expert']:
        train_expert_policy(config)
    print()
    dump(config)

    gen_data_from_expert(config)
    print()
    dump(config)

    do_dagger(config)
    print()
    dump(config)

    print('Done with all steps')
    print('total time taken = %f' % (timer.time() - ts))

def dump(config):
    config_file = os.path.join(config['main_dir'], 'config.json')

    with open(config_file, 'w') as fp:
        json.dump(config, fp)


if __name__ == '__main__':
    config = DEFAULT_CONFIG
    config['main_dir'] = os.path.join(DATA_DIR, '%s_%s' % (config['env_name'], config['id_post']))
    ensure_dir(config['main_dir'])
    config['id'] = '%s_id_%s' % (config['env_name'], config['id_post'])
    config['env_id'] = ENV_ID[config['env_name']]

    main(config)


DEFAULT_CONFIG = dict(
    id_post='test_hammer_less_n_traj',
    beta_decay= 0.2,
    beta_start=1.0,
    num_traj_expert= 5,
    dagger_epoch= 50,
    horizon_il= 150,
    has_robot_info= True,
    train_traj_per_file= 5,
    gen_traj_dagger_ep= 5,
    env_name= "hand_hammer",
    delta= 0.01,
    sliding_window= 100,
    seed= 1000,
    expert_policy_folder= "hand_hammer_expert",
    traj_budget_expert= 12500,
    lr= 0.0003,
    use_tactile= False,
    trainer_epochs= 100,
    eval_num_traj= 30,
    num_files_train= 1,
    batch_size_viz_pol= 128,
    val_traj_per_file= 5,
    num_files_val= 1,
    use_cuda=True,
    camera_name='vil_camera',
    train_expert=False,
    device_id=0,
    use_late_fusion=True,
)