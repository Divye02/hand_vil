from mjrl_mod.envs import mujoco_env
from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0
import numpy as np

class HandHammer(HammerEnvV0):
    def __init__(self):
        self.use_tactile=True
        HammerEnvV0.__init__(self)
        self.robot_info_dim = len(self._get_robot_specific_obs())

    def _step(self, a):
        o, r, t, info = super()._step(a)
        robot_info = self._get_robot_specific_obs()
        info['robot_info'] = robot_info
        return o, r, t, info

    def _get_robot_specific_obs(self):
        # return self._get_obs()
        robot_jnt = self.data.qpos.ravel()[:-6]
        robot_vel = self.data.qvel.ravel()[:-6]
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        sensordata = []
        if self.use_tactile:
            sensordata = self.data.sensordata.ravel().copy()[:41]
            sensordata = np.clip(sensordata, -5.0, 5.0)

        res = np.concatenate([robot_jnt, robot_vel, palm_pos, sensordata])
        self.robot_info_dim = len(res)
        return res

    def reset_model(self):
        o = super().reset_model()
        robot_info = self._get_robot_specific_obs()
        return o, dict(robot_info=robot_info)

    def get_body_com(self, *args, **kwargs):
        return mujoco_env.MujocoEnv.get_body_com(self, *args, **kwargs)

    def get_pixels(self, *args, **kwargs):
        return mujoco_env.MujocoEnv.get_pixels(self, *args, **kwargs)
   
    def visualize_policy(self, *args, **kwargs):
        mujoco_env.MujocoEnv.visualize_policy(self, *args, **kwargs)

    def visualize_policy_offscreen(self, *args, **kwargs):
        mujoco_env.MujocoEnv.visualize_policy_offscreen(self, *args, **kwargs)

    def init(self, *args, **kwargs):
        return mujoco_env.MujocoEnv.init(self, *args, **kwargs)

    def single_step(self, *args, **kwargs):
        return mujoco_env.MujocoEnv.single_step(self, *args, **kwargs)
