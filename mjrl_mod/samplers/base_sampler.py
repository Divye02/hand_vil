import logging
from tqdm import tqdm

from settings import FRAME_SIZE

logging.disable(logging.CRITICAL)

import numpy as np
from mjrl_mod.utils.get_environment import get_environment
from mjrl_mod.utils import tensor_utils

# Single core rollout to sample trajectories
# =======================================================
def do_rollout(N,
    policy,
    T=1e6,
    env=None,
    env_name=None,
    pegasus_seed=None,
    use_mean=False,
    save_img=False,
    camera_name=None,
    device_id=None):
    """
    params:
    N               : number of trajectories
    policy          : policy to be used to sample the data
    T               : maximum length of trajectory
    env             : env object to sample from
    env_name        : name of env to be sampled from 
                      (one of env or env_name must be specified)
    pegasus_seed    : seed for environment (numpy speed must be set externally)
    """

    if env_name is None and env is None:
        print("No environment specified! Error will be raised")
    if env is None: env = get_environment(env_name)
    if pegasus_seed is not None: env.env.env._seed(pegasus_seed)
    T = min(T, env.horizon) 

    #print("####### Worker started #######")
    
    paths = []

    for ep in tqdm(range(N)):

        # Set pegasus seed if asked
        if pegasus_seed is not None:
            seed = pegasus_seed + ep
            env.env.env._seed(seed)
            np.random.seed(seed)
        else:
            np.random.seed()
        
        observations=[]

        all_robot_info = []
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []
        path_image_pixels = []

        o = env.reset()
        if env.has_robot_info_attr():
            o, env_state = o
            robot_info = env_state['robot_info']
        done = False
        t = 0

        while t < T and done != True:
            a, agent_info = policy.get_action(o)
            if use_mean:
                a = agent_info['evaluation']

            if save_img:
                image_pix = env.get_pixels(frame_size=FRAME_SIZE, camera_name=camera_name, device_id=device_id)
            next_o, r, done, env_info = env.step(a)
            #observations.append(o.ravel())
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            if save_img:
                path_image_pixels.append(image_pix)

            if env.has_robot_info_attr():
                all_robot_info.append(robot_info)
                robot_info = env_info['robot_info']


            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done,
        )
        if save_img:
            path['image_pixels'] = np.array(path_image_pixels)
        if env.has_robot_info_attr():
            path['robot_info'] = np.array(all_robot_info)

        paths.append(path)

    #print("====== Worker finished ======")
    del(env)
    return paths

def do_rollout_star(args_list):
    return do_rollout(*args_list)
