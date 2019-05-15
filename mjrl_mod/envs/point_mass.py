import numpy as np
from gym import utils
from mjrl_mod.envs import mujoco_env
from mjrl.envs.point_mass import PointMassEnv

class PointMassEnv(PointMassEnv, mujoco_env.MujocoEnv):
    def __init__(self):
        self.agent_bid = 0
        self.target_sid = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, '/Users/divye/Documents/research/vil_paper/visuomotor-hand-man/mjrl_mod/envs/assets/point_mass.xml', 5)
        self.agent_bid = self.sim.model.body_name2id('agent')
        self.target_sid = self.sim.model.site_name2id('target')

        self.robot_info_dim = len(self._get_robot_specific_obs())

    def _step(self, a):
        o, r, t, info = super()._step(a)
        robot_info = self._get_robot_specific_obs()
        info['robot_info'] = robot_info
        return o, r, t, info

    def _get_robot_specific_obs(self):
        # return self._get_obs()
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()[:2]
        agent_vel = self.data.qvel.ravel();
        target_pos = self.data.site_xpos[self.target_sid].ravel()[:2]
        # So that we can add other things, if we need to.
        res = np.concatenate([agent_pos, agent_vel, target_pos])
        self.robot_info_dim = len(res)
        return res

    def reset_model(self):
        o = super().reset_model()
        robot_info = self._get_robot_specific_obs()
        return o, dict(robot_info=robot_info)

    def visualize_policy(self, *args, **kwargs):
        mujoco_env.MujocoEnv.visualize_policy(self, *args, **kwargs)

    def visualize_policy_offscreen(self, *args, **kwargs):
        mujoco_env.MujocoEnv.visualize_policy_offscreen(self, *args, **kwargs)
