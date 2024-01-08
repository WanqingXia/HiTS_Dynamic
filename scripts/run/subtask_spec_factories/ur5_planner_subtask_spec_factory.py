import numpy as np

from graph_rl.graph_rl.spaces import BoxSpace

from .box_subtask_spec_factory import BoxSubtaskSpecFactory


class UR5PlannerSubtaskSpecFactory(BoxSubtaskSpecFactory):

    @classmethod
    def bound_angle(cls, angles):
        for num, angle in enumerate(angles):
            if num == 2:
                bounded_angle = np.absolute(angle) % np.pi
            else:
                bounded_angle = np.absolute(angle) % (2 * np.pi)

            angles[num] = -bounded_angle if angle <0 else bounded_angle

        return angles

    @classmethod
    def get_indices_and_factorization(cls, env, subtask_spec_params, level):
        """Determines observation and goal space and factorization of goal space.
        
        Returns
        partial_obs_indices: Indices of components of "observation" item of
            observation space that are exposed to the level.
        goal_indices: Indices of components of "observation" item of
            observation space that comprise the goal space.
        factorization: List of lists of indices which define the subspaces
            of the goal space in which the Euclidean distance is used.
        """
        # nobs = 29: ee_pos=3, ee_ori=3, joint_pos=6,
        # obs_pos=3, obs_ori=3, obs_vel=3, obs_ang_vel=3, link_dist=5
        n_obs = env.observation_space["observation"].shape[0]
        partial_obs_indices = range(12)
        # position of goal_pos and goal_ori in observation
        goal_indices = range(6, 12)
        # same as in original HAC paper
        factorization = [[i] for i in range(6)]
        return partial_obs_indices, goal_indices, factorization

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""
        # method of the highest level
        def mapping(partial_obs):
            return np.array([cls.bound_angle(partial_obs[6:])])
        return mapping

    @classmethod
    def get_map_to_subgoal_and_subgoal_space(cls, env):
        """Return map from partial observation to subtask achieved goal."""

        def mapping(partial_obs):
            return partial_obs[:6]

        # Define the range for subgoal (top level action space)
        def mapping(partial_obs):
            # joint angles are 6-12 in partial_obs
            angles = np.array([cls.bound_angle(partial_obs[6:])])
            return angles

        high = np.array([2.*np.pi]*6)
        high[2] = np.pi  # the elbow joint is limited to [-pi, pi]
        low = -high
        subgoal_space = BoxSpace(low, high, dtype=np.float32)
        return mapping, subgoal_space
