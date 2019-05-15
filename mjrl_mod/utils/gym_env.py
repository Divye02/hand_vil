import gym
import numpy as np
from tqdm import tqdm

from settings import FRAME_SIZE


class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon, num_agents):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon
        self.num_agents = num_agents

    def as_dict(self):
        return dict(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            horizon=self.horizon,
            num_agents=self.num_agents
        )


class GymEnv(object):
    def __init__(self, env_name, use_tactile=True):
        env = gym.make(env_name)
        print('Using tactile:', use_tactile)
        env.env.use_tactile = use_tactile
        env.env.robot_info_dim = len(env.env._get_robot_specific_obs())

        self.env = env
        self.env_id = env.spec.id

        self._horizon = env.spec.timestep_limit
        try:
            self._action_dim = self.env.env.action_dim
        except AttributeError:
            self._action_dim = self.env.env.action_space.shape[0]

        self._observation_dim = self.env.env.obs_dim

        try:
            self._num_agents = self.env.env.num_agents
        except AttributeError:
            self._num_agents = 1

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon, self._num_agents)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def has_robot_info_attr(self):
        return hasattr(self.env.env, 'robot_info_dim')

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration', use_img=False, use_seq=False,
                        frame_size=FRAME_SIZE, camera_name=None):
        self.env.env.visualize_policy(policy, horizon, num_episodes, mode, use_img, use_seq,
                                      has_robot_info=self.has_robot_info_attr(), frame_size=frame_size,
                                      camera_name=camera_name)

    def visualize_policy_offscreen(self, save_loc, filename, policy, horizon=1000, num_episodes=1, mode='exploration',
                                   use_img=False, camera_name="", use_seq=False, pickle_dump=False):
        self.env.env.visualize_policy_offscreen(policy, horizon, num_episodes, mode=mode, save_loc=save_loc,
                                                filename=filename, use_img=use_img, camera_name=camera_name,
                                                use_seq=use_seq, pickle_dump=pickle_dump,
                                                has_robot_info=self.has_robot_info_attr())

    def get_pixels(self, frame_size=FRAME_SIZE, camera_name=None, device_id=0):
        return self.env.env.get_pixels(frame_size, camera_name, device_id)

    def visualize_policy_data(self, data):
        self.env.env.visualize_policy_data(data)

    def visualize_policy_data_single(self, data):
        self.env.env.visualize_policy_data_single(data)

    def evaluate_policy(self, policy,
                        num_episodes=5,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=[],
                        get_full_dist=False,
                        mean_action=False,
                        init_state=None,
                        terminate_at_done=True,
                        save_video_location=None,
                        seed=None,
                        use_seq=False,
                        camera_name=None,
                        device_id=None,
                        use_cuda=False):

        if seed is not None:
            self.env.env._seed(seed)
            np.random.seed(seed)
        horizon = self._horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)

        if save_video_location != None:
            self.env.monitor.start(save_video_location, force=True)

        for ep in tqdm(range(num_episodes)):

            if init_state is not None:
                o = self.reset()
                self.env.set_state(init_state[0], init_state[1])
                o = self.env._get_obs()
            else:
                o = self.reset()

            robot_info = None
            if self.has_robot_info_attr():
                o, env_info = self.env.reset()
                robot_info = env_info['robot_info']

            path_image_pixels = []
            t, done = 0, False
            while t < horizon and not (done and terminate_at_done):
                if visual == True:
                    self.render()

                if use_seq:
                    image_pix = self.get_pixels(frame_size=FRAME_SIZE, camera_name=camera_name, device_id=device_id)
                    img = image_pix
                    prev_img = image_pix
                    prev_prev_img = image_pix
                    if t > 0:
                        prev_img = path_image_pixels[t - 1]

                    if t > 1:
                        prev_prev_img = path_image_pixels[t - 2]
                    path_image_pixels.append(img)
                    prev_prev_img = np.expand_dims(prev_prev_img, axis=0)
                    prev_img = np.expand_dims(prev_img, axis=0)
                    img = np.expand_dims(img, axis=0)

                    o = np.concatenate((prev_prev_img, prev_img, img), axis=0)

                    if mean_action:
                        a = policy.get_action(o, robot_info=robot_info, use_cuda=use_cuda)[1]['mean']
                    else:
                        a = policy.get_action(o, robot_info=robot_info, use_cuda=use_cuda)[0]
                else:
                    if mean_action:
                        a = policy.get_action(o)[1]['mean']
                    else:
                        a = policy.get_action(o)[0]

                o, r, done, env_info = self.step(a)

                if self.has_robot_info_attr():
                    robot_info = env_info['robot_info']

                ep_returns[ep] += (gamma ** t) * r
                t += 1

        if save_video_location != None:
            self.env.monitor.close()

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        full_dist = []

        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        if get_full_dist == True:
            full_dist = ep_returns

        return [base_stats, percentile_stats, full_dist]
