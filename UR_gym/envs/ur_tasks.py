import numpy as np

from UR_gym.envs.core import RobotTaskEnv
from UR_gym.envs.robots.UR5 import UR5Ori
from UR_gym.envs.tasks.reach import ReachDyn
from UR_gym.pyb_setup import PyBullet

class UR5DynReachEnv(RobotTaskEnv):
    """Reach task wih UR5 robot. (Added obstacle reward)

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
    """

    def __init__(self, render: bool = False) -> None:
        sim = PyBullet(render=render)
        robot = UR5Ori(sim, block_gripper=True, base_position=np.array([0.0, 0.0, 0.0]))
        task = ReachDyn(sim, robot=robot)
        super().__init__(robot, task)
