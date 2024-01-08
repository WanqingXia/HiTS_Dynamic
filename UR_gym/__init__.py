import os
from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()


register(
    id="UR5DynReach-v1",
    entry_point="UR_gym.envs:UR5DynReachEnv",
    max_episode_steps=100,
)
