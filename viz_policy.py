import pickle
from mjrl_mod.utils.gym_env import GymEnv
from settings import *
import mjrl_mod.envs
# import solveHMS.envs
# import mj_envs.hand_manipulation_suite

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

CAMERA_NAME = {
    'cheetah': 'track_cam',
    'ant': 'track_cam',
    'reacher': 'front_view',
    'point_mass': 'top_view',
    # Add  Other envs here
    'hand_pickup': 'vil_camera',
    'hand_hammer': 'vil_camera',
    'hand_pen': 'view_5',
    'hand_door': 'vil_camera'
}

def main():



    # cam_name = 'view5'


    # id = 'mjrl_half_cheetah-v0'
    # e = GymEnv('mjrl_point_mass-v0', use_tactile=True)
    # e = GymEnv('mjrl_hand_hammer-v0', use_tactile=False)

    env_name = 'hand_pen'
    # env_id = 'mjrl_hand_hammer_viz-v0'
    # env_id = 'mjrl_hand_door_viz-v0'
    # env_id = 'mjrl_hand_pickup_viz-v0'
    env_id = 'mjrl_hand_pen_viz-v0'
    viz_folder = 'final_pen_v5_less_traj'
    # Cam name in the dict above

    print('Visualizing %s' % env_name)
    # env_id = 'mjrl_hand_door_viz-v0'
    # env_id = 'mjrl_hand_pen_viz-v0'
    # env_id = 'mjrl_hand_pickup_viz-v0'
    e = GymEnv(env_id, use_tactile=True)

    # e = GymEnv('SHAP_slide_pickup-v42', use_tactile=False)
    # e = GymEnv('pen_reposition-v2', use_tactile=False)
    # p_p = '/Users/divye/Documents/research/vil_paper/visuomotor-hand-man/expert_policies/hammer.pickle'
    p_p = '/Users/divye/Documents/research/vil_paper/visuomotor-hand-man/temp_pols/trained_policy_ep_10.pickle_pen_v5_lesstraj'
    policy = pickle.load(open(p_p, 'rb'))
    print('usind %d horizon', e.horizon)
    policy.model.eval()
    policy.old_model.eval()
    # reward, _, _ = e.evaluate_policy(policy,
    #                     num_episodes=10, mean_action=True, use_seq=True,
    #                     camera_name=CAMERA_NAME[env_name], seed=500)

    # print('reward', reward)

    # e.visualize_policy(policy, num_episodes=20, horizon=50, mode='evaluation', use_img=True, use_seq=True, frame_size=FRAME_SIZE, camera_name=cam_name)
    e.visualize_policy_offscreen(ensure_dir(os.path.join(VIDOES_FOLDER, viz_folder)) + '/', env_name, policy=policy, num_episodes=3, horizon=e.horizon, mode='evaluation', use_img=True, use_seq=True, camera_name=CAMERA_NAME[env_name], pickle_dump=False)
    del(e)

if __name__ == '__main__':
    main()


