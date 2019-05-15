#
from gym.envs.registration import register
register(
    id='mjrl_hand_hammer_viz-v0',
    entry_point='mjrl_mod.envs:HandHammer',
    max_episode_steps=250,
)
register(
    id='mjrl_hand_door_viz-v0',
    entry_point='mjrl_mod.envs:HandDoor',
    max_episode_steps=250,
)
register(
    id='mjrl_hand_pen_viz-v0',
    entry_point='mjrl_mod.envs:HandPen',
    max_episode_steps=250,
)
register(
    id='mjrl_hand_pickup_viz-v0',
    entry_point='mjrl_mod.envs:HandPickup',
    max_episode_steps=400,
)
from mj_envs.mujoco_env import MujocoEnv
from mjrl_mod.envs.hand_hammer import HandHammer
from mjrl_mod.envs.hand_door import HandDoor
from mjrl_mod.envs.hand_pen import HandPen
from mjrl_mod.envs.hand_pickup import HandPickup
