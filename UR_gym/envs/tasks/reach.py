from typing import Any, Dict
import os
import numpy as np
from UR_gym.envs.core import Task
from UR_gym.utils import *


class ReachDyn(Task):
    def __init__(
        self,
        sim,
        robot,
    ) -> None:
        super().__init__(sim)
        self.robot = robot
        self.goal_range_low = np.array([0.3, -0.5, 0.0])  # table width, table length, height
        self.goal_range_high = np.array([0.75, 0.5, 0.2])
        self.obs_range_low = np.array([0.5, -0.5, 0.25])  # table width, table length, height
        self.obs_range_high = np.array([1.0, 0.5, 0.55])

        # margin and weight
        self.distance_threshold = 0.05  # 5cm
        self.ori_distance_threshold = 0.0873  # 5 degrees
        self.action_weight = -1
        self.collision_weight = -500
        self.distance_weight = -70
        self.orientation_weight = -30
        link_weights = [8, 2.4, 1.2, 1.2, 0.2]
        self.dist_change_weight = np.array(link_weights) / np.sum(link_weights) * 50
        self.success_weight = 200

        # Stored values
        self.obstacle_start = np.zeros(6)
        self.obstacle_end = np.zeros(6)
        self.velocity = np.zeros(6)
        self.collision = False
        self.link_dist = np.zeros(5)
        self.last_dist = np.zeros(5)
        self.step_num = 0

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-1.04)
        self.sim.create_table(length=1.1, width=1.8, height=0.92, x_offset=0.5, z_offset=-0.12)
        self.sim.create_track(length=0.2, width=1.1, height=0.12, x_offset=0.0, z_offset=0.0)
        self.sim.create_target(
            body_name="target",
            half_extents=np.ones(3) * 0.05 / 2,
            mass=0.0,
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 1.0]),
            texture=os.getcwd() + "/UR_gym/assets/colored_cube_ori.png",  # the robot end-effector should point to blue
        )
        self.sim.create_cylinder(
            body_name="obstacle",
            radius=0.05,
            height=0.4,
            mass=0.0,
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([1.0, 0.92, 0.8, 1.0]),
            texture=os.getcwd() + "/UR_gym/assets/cylinder.png",
        )
        self.sim.create_target(
            body_name="subgoal_target",
            half_extents=np.ones(3) * 0.05 / 2,
            mass=0.0,
            ghost=False,
            position=np.array([0.5, 0.0, 1.0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.5]),
            texture=os.getcwd() + "/UR_gym/assets/colored_cube_ori.png",  # the robot end-effector should point to blue
        )

        self.sim.create_box(
            body_name="zone_goal",
            half_extents=np.array([0.225, 0.5, 0.1]),
            mass=0.0,
            ghost=True,
            position=np.array([0.525, 0.0, 0.1]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
        )
        self.sim.create_box(
            body_name="zone_obs",
            half_extents=np.array([0.25, 0.5, 0.15]),
            mass=0.0,
            ghost=True,
            position=np.array([0.75, 0.0, 0.4]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.2]),
        )

    def get_obs(self) -> np.ndarray:
        obstacle_position = self.sim.get_base_position("obstacle")
        obstacle_rotation = self.sim.get_base_rotation("obstacle")
        obstacle_current = np.concatenate((obstacle_position, obstacle_rotation))
        return np.concatenate((obstacle_current, self.velocity, self.link_dist))

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        ee_orientation = np.array(self.robot.get_ee_orientation())
        return np.concatenate((ee_position, ee_orientation))

    def reset(self) -> None:
        self.collision = False
        self.step_num = 0
        distance_fail = True
        while distance_fail:
            self.goal = self._sample_goal()
            self.obstacle_start = self._sample_obstacle()
            self.obstacle_end = self._sample_obstacle()
            self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])
            self.sim.set_base_pose("obstacle", self.obstacle_end[:3], self.obstacle_end[3:])
            start_end_dist = distance(self.obstacle_end, self.obstacle_start)
            distance_fail = (self.sim.get_target_to_obstacle_distance() < 0.1) or (start_end_dist < 0.3)

        # set obstacle to start position after checking
        self.sim.set_base_pose("obstacle", self.obstacle_start[:3], self.obstacle_start[3:])
        self.collision = self.sim.check_collision()
        self.link_dist = self.sim.get_link_distances()
        self.last_dist = self.link_dist
        if self.collision:
            print("Collision after reset, this should not happen")

    def set_subgoal(self, pose):
        self.sim.set_base_pose("subgoal_target", pose[:3], pose[3:])

    def set_goal_and_obstacle(self, test_data):
        self.goal = test_data[:6]
        self.obstacle_start = test_data[6:12]
        self.obstacle_end = test_data[12:]

        # set the rotation for obstacle to same value for linear movement
        self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])
        # set final pose to keep the environment constant
        self.sim.set_base_pose("obstacle", self.obstacle_start[:3], self.obstacle_start[3:])
        self.collision = self.sim.check_collision()
        self.link_dist = self.sim.get_link_distances()
        self.last_dist = self.link_dist

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high))
        goal_rot = sample_euler_constrained()
        goal = np.concatenate((goal_pos, goal_rot))
        return goal

    def _sample_obstacle(self):
        obstacle_pos = self.np_random.uniform(self.obs_range_low, self.obs_range_high)
        obstacle_rot = sample_euler_obstacle()
        obstacle = np.concatenate((obstacle_pos, obstacle_rot))
        return obstacle

    def set_velocity(self):
        """
        This function is used to control the movement of obstacle
        The obstacle moves in a constant speed from start pose to end pose
        When end pose is achieved, speed is set to zero
        """

        if self.step_num < 25:
            time_duration = 2
            linear_velocity = (self.obstacle_end[:3] - self.obstacle_start[:3]) / time_duration
            # Calculating the relative rotation from start to end orientation
            rot_end = self.sim.euler_to_quaternion(self.obstacle_end[3:])
            rot_start = self.sim.euler_to_quaternion(self.obstacle_start[3:])
            relative_rotation = self.sim.get_quaternion_difference(rot_start, rot_end)
            # Convert the relative rotation quaternion to axis-angle representation
            axis, angle = self.sim.get_axis_angle(relative_rotation)
            # Calculate the angular velocity required to achieve the rotation in 1 second
            angular_velocity = np.array(axis) * angle / time_duration
            self.sim.set_velocity("obstacle", linear_velocity, angular_velocity)
            self.velocity = np.concatenate((linear_velocity, angular_velocity))
        else:
            linear_velocity = np.zeros(3)
            angular_velocity = np.zeros(3)
            self.sim.set_velocity("obstacle", linear_velocity, angular_velocity)
            self.velocity = np.concatenate((linear_velocity, angular_velocity))
        self.step_num += 1

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        distance_success = distance(achieved_goal, desired_goal) < self.distance_threshold
        orientation_success = angular_distance(achieved_goal, desired_goal) < self.ori_distance_threshold
        return np.array(distance_success & orientation_success, dtype=np.bool8)

    def check_collision(self) -> bool:
        self.collision = self.sim.check_collision()
        return self.collision

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        # Actions made by the top level is the goal for bottom level
        # The reward here should evaluate the quality of the action,
        # otherwise the bottom level will not get meaningful goal

        """"collision reward"""
        if self.collision:
            return np.float64(self.collision_weight)
        """success reward"""
        if self.is_success(achieved_goal, desired_goal):
            return np.float64(self.success_weight)

        self.link_dist = self.sim.get_link_distances()
        dist_change = self.link_dist - self.last_dist
        self.last_dist = self.link_dist
        dist = distance(achieved_goal, desired_goal)
        ori_dist = angular_distance(achieved_goal, desired_goal)

        reward = np.float64(0.0)

        """distance reward"""
        reward += self.distance_weight * dist
        """orientation reward"""
        reward += self.orientation_weight * ori_dist
        """obstacle distance reward"""
        reward_changes = np.where(self.link_dist < 0.2, self.dist_change_weight * dist_change, 0)
        reward += reward_changes.sum()
        return reward
