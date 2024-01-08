from typing import Optional
import os
import numpy as np
from gymnasium import spaces

from UR_gym.envs.core import PyBulletRobot
from UR_gym.pyb_setup import PyBullet

class UR5Ori(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to True.
        base_position (np.ndarray, optionnal): Position of the base of the robot, as (x, y, z). Defaults to (0, 0, 0).
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = True,
        base_position: Optional[np.ndarray] = None,
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.action = np.zeros(6)
        action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="UR5",
            file_name=os.getcwd() + "/UR_gym/envs/robots/urdf/ur5e.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([1, 2, 3, 4, 5, 6]),  # 1-6: robot joints, 9: gripper finger
            joint_forces=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0]),  # may need to add all joint fingers later
        )

        self.neutral_joint_values = np.array([0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0])
        self.ee_link = 7  # the id of ee_link
        self.block_gripper = block_gripper

        # following parameters are for OMPL
        self.id = 0
        self.joint_idx = np.array([1, 2, 3, 4, 5, 6])
        self.num_dim = len(self.joint_idx)
        self.joint_bounds = []
        self.state = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.action = action[:6] * np.pi  # map joint velocity from -1~+1 to -pi~+pi
        target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(self.action)
        # self.control_joints(target_angles=target_arm_angles)
        self.set_joint_angles(angles=target_arm_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """

        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=ee_displacement[:3], orientation=np.roll(ee_displacement[3:], -1))
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 6 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.1  # limit maximum change in position, 0.3 rad everytime
        # get the current position and the target position
        current_arm_joint_angles = np.array(self.get_joint_angles())
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position, orientation and velocity
        ee_position = np.array(self.get_ee_position())
        ee_orientation = np.array(self.get_ee_orientation())
        joint_angles = np.array(self.get_joint_angles())
        return np.concatenate((ee_position, ee_orientation, joint_angles))

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_orientation(self) -> np.ndarray:
        """Returns the orientation of the end-effector as euler(x, y, z)"""
        return self.get_link_orientation(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def get_joint_angles(self) -> np.ndarray:
        """Returns the angles of the all 6 joints as (j1, j2, j3, j4, j5, j6)"""
        angles = np.zeros(6)
        for i, ind in enumerate(self.joint_indices):
            angles[i] = self.get_joint_angle(ind)
        return angles

    def get_action(self):
        """Returns the action of the all 6 joints as (a1, a2, a3, a4, a5, a6)"""
        return self.action

    # following functions are for OMPL
    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = self.sim.physics_client.getJointInfo(self.id, joint_id)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds
            if low < high:
                self.joint_bounds.append([low, high])
        print("Joint bounds: {}".format(self.joint_bounds))
        return self.joint_bounds

    def get_cur_state(self):
        return self.get_joint_angles()

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self.set_joint_angles(state)
        self.state = state

