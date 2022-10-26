from copy import copy
import math
import os

import numpy as np
import hydra
from typing import Callable

from omegaconf import OmegaConf
import isaacgym
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from sympy import print_fcode
import torch

from dexgrasp.utils.isaacgym_math import quaternion_mul
from dexgrasp.utils.isaacgym_drawing import draw_6D_pose, draw_bbox


class KukaAllegroGrasp(VecTask):
    """VecTask-like API, only joint control"""

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        # Param initialization
        self.cfg = cfg

        # Reward related
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.control_mode = self.cfg["env"]["controlMode"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Action parameters
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        # Task params
        self.num_objects = self.cfg["env"]["numObjects"]
        self.box_size = 0.05
        self.object_type = self.cfg["env"]["objectType"]
        self.goal_random_range = self.cfg["task"]["goal"]["random_range"]

        # Env settings
        self.cfg["env"]["numObservations"] = 42
        self.cfg["env"]["numStates"] = 42
        self.cfg["env"]["numActions"] = 23

        # Init VecTask
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # Set initial camera pose
        cam_pos = gymapi.Vec3(2.0, 1.5, 0.0)
        cam_target = gymapi.Vec3(0.0, 0.5, 0.0)
        # Because this is a vector environment, we need to determine the origin
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Get GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Create some wrapper tensors for different slices
        self.num_kuka_dofs = 7 + 16
        self.kuka_default_dof_pos = torch.zeros(
            self.num_kuka_dofs, dtype=torch.float, device=self.device
        )
        # Set the default dof positions
        self.kuka_default_dof_pos[0] = -0.18
        self.kuka_default_dof_pos[1] = -0.25
        self.kuka_default_dof_pos[2] = 0.09
        self.kuka_default_dof_pos[3] = 0.96
        self.kuka_default_dof_pos[4] = 0.30
        self.kuka_default_dof_pos[5] = -0.83
        self.kuka_default_dof_pos[6] = -1.12

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.kuka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_kuka_dofs
        ]
        self.kuka_dof_pos = self.kuka_dof_state[..., 0]
        self.kuka_dof_vel = self.kuka_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )

        # Reward-related
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device
        )

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # Commonly used tensors
        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # If randomizing, apply once immediately on startup before the fist self.sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        if self.sim_params.up_axis == gymapi.UP_AXIS_Z:
            # Z-axis up
            plane_params.normal = gymapi.Vec3(0, 0, 1)
        else:
            # Y-axis up
            plane_params.normal = gymapi.Vec3(0, 1, 0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # Configuration
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        # Configure table
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        table_dims = gymapi.Vec3(0.6, 0.4, 1.0)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.7, 0.5 * table_dims.y + 0.001, 0.0)
        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options
        )

        # Configure & load objects
        asset_options.fix_base_link = False

        can_asset_file = self.cfg["env"]["asset"]["assetFileNameCan"]
        banana_asset_file = self.cfg["env"]["asset"]["assetFileNameBanana"]
        mug_asset_file = self.cfg["env"]["asset"]["assetFileNameMug"]
        brick_asset_file = self.cfg["env"]["asset"]["assetFileNameBrick"]
        object_files = []
        object_files.append(can_asset_file)
        object_files.append(banana_asset_file)
        object_files.append(mug_asset_file)
        object_files.append(brick_asset_file)

        object_assets = []
        object_assets.append(
            self.gym.create_box(
                self.sim, self.box_size, self.box_size, self.box_size, asset_options
            )
        )
        object_assets.append(
            self.gym.load_asset(self.sim, asset_root, can_asset_file, asset_options)
        )
        object_assets.append(
            self.gym.load_asset(self.sim, asset_root, banana_asset_file, asset_options)
        )
        object_assets.append(
            self.gym.load_asset(self.sim, asset_root, mug_asset_file, asset_options)
        )
        object_assets.append(
            self.gym.load_asset(self.sim, asset_root, brick_asset_file, asset_options)
        )

        spawn_height = gymapi.Vec3(0.0, 0.0, 0.0)

        # Configure & load kuka
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True

        kuka_asset_file = self.cfg["env"]["asset"]["assetFileNameKuka"]
        kuka_asset = self.gym.load_asset(
            self.sim, asset_root, kuka_asset_file, asset_options
        )
        print(f"Loading asset {kuka_asset_file} from {asset_root}")

        # Configure kuka actuators
        # Get DOF props
        self.num_kuka_bodies = self.gym.get_asset_rigid_body_count(kuka_asset)
        self.num_kuka_shapes = self.gym.get_asset_rigid_shape_count(kuka_asset)
        self.num_kuka_dofs = self.gym.get_asset_dof_count(kuka_asset)

        self.actuated_dof_indices = [i for i in range(self.num_kuka_dofs)]
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_kuka_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_kuka_dofs), dtype=torch.float, device=self.device
        )

        # Set kuka dof properties
        self.kuka_dof_lower_limits = []
        self.kuka_dof_upper_limits = []
        self.kuka_dof_default_pos = []
        self.kuka_dof_default_vel = []
        self.sensors = []
        kuka_dof_props = self.gym.get_asset_dof_properties(kuka_asset)

        for i in range(self.num_kuka_dofs):
            self.kuka_dof_lower_limits.append(kuka_dof_props["lower"][i])
            self.kuka_dof_upper_limits.append(kuka_dof_props["upper"][i])
            self.kuka_dof_default_pos.append(0.0)
            self.kuka_dof_default_vel.append(0.0)
            if i in self.actuated_dof_indices:
                if self.control_mode == "position":
                    kuka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
                elif self.control_mode == "velocity":
                    kuka_dof_props["driveMode"][i] = gymapi.DOF_MODE_VEL
                elif self.control_mode == "torque":
                    kuka_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
                kuka_dof_props["effort"][i] = 0.5
                kuka_dof_props["stiffness"][i] = 3
                kuka_dof_props["damping"][i] = 0.1
                kuka_dof_props["friction"][i] = 0.01
                kuka_dof_props["armature"][i] = 0.001
            else:
                # None control for arm
                kuka_dof_props["driveMode"][i] = gymapi.DOF_MODE_NONE

        # Get joint limits and ranges for kuka
        self.actuated_dof_indices = to_torch(
            self.actuated_dof_indices, dtype=torch.long, device=self.device
        )
        self.kuka_dof_lower_limits = to_torch(
            self.kuka_dof_lower_limits, device=self.device
        )
        self.kuka_dof_upper_limits = to_torch(
            self.kuka_dof_upper_limits, device=self.device
        )
        self.kuka_dof_default_pos = to_torch(
            self.kuka_dof_default_pos, device=self.device
        )
        self.kuka_dof_default_vel = to_torch(
            self.kuka_dof_default_vel, device=self.device
        )

        # Cache some common handles for later use
        self.envs = []
        self.kuka_handles = []
        self.kuka_indices = []
        self.tray_handles = []
        self.object_handles = []
        self.object_indices = []

        # Goal state cache
        self.goal_states = torch.zeros(
            (self.num_envs, 13), dtype=torch.float, device=self.device
        )
        self.goal_random_center = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )

        # Color initialization
        colors = [
            gymapi.Vec3(1.0, 0.0, 0.0),
            gymapi.Vec3(1.0, 127.0 / 255.0, 0.0),
            gymapi.Vec3(1.0, 1.0, 0.0),
            gymapi.Vec3(0.0, 1.0, 0.0),
            gymapi.Vec3(0.0, 0.0, 1.0),
            gymapi.Vec3(39.0 / 255.0, 0.0, 51.0 / 255.0),
            gymapi.Vec3(139.0 / 255.0, 0.0, 1.0),
        ]
        tray_color = gymapi.Vec3(0.24, 0.35, 0.8)
        banana_color = gymapi.Vec3(0.85, 0.88, 0.2)
        brick_color = gymapi.Vec3(0.9, 0.5, 0.1)

        # Create environment grid
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(self.num_envs))

        print(f"Creating {self.num_envs} environments.")
        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # Add table
            table_handle = self.gym.create_actor(
                env, table_asset, table_pose, "table", i, 0
            )

            table_corner = table_pose.p - table_dims * 0.5
            x = table_corner.x + table_dims.x * 0.5
            y = table_dims.y + self.box_size + 0.01
            z = table_corner.z + table_dims.z * 0.5

            # Add kuka
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

            kuka_handle = self.gym.create_actor(env, kuka_asset, pose, "kuka", i, 1)
            kuka_indice = self.gym.get_actor_index(env, kuka_handle, gymapi.DOMAIN_SIM)
            self.kuka_handles.append(kuka_handle)
            self.kuka_indices.append(kuka_indice)
            self.gym.set_actor_dof_properties(env, kuka_handle, kuka_dof_props)

            # Add objects & goal center
            self.goal_random_center[i, 0] = table_corner.x + table_dims.x * 0.5
            self.goal_random_center[i, 1] = (
                table_corner.y + table_dims.y + self.goal_random_range[1] + 0.05
            )
            self.goal_random_center[i, 2] = table_corner.z + table_dims.z * 0.5

            object_pose = gymapi.Transform()
            for j in range(self.num_objects):
                x = table_corner.x + table_dims.x * 0.5 + np.random.rand() * 0.35 - 0.2
                y = table_dims.y + self.box_size * 1.2 * j - 0.05
                z = table_corner.z + table_dims.z * 0.5 + np.random.rand() * 0.3 - 0.15

                object_pose.p = gymapi.Vec3(x, y, z) + spawn_height

                object_asset = object_assets[0]
                if self.object_type >= 5:
                    object_asset = object_assets[np.random.randint(len(object_assets))]
                else:
                    object_asset = object_assets[self.object_type]

                object_handle = self.gym.create_actor(
                    env, object_asset, object_pose, "object" + str(j), i, 0
                )
                self.object_handles.append(object_handle)
                object_idx = self.gym.get_actor_index(
                    env, object_handle, gymapi.DOMAIN_SIM
                )
                self.object_indices.append(object_idx)

                if self.object_type == 2:
                    color = gymapi.Vec3(
                        banana_color.x + np.random.rand() * 0.1,
                        banana_color.y + np.random.rand() * 0.05,
                        banana_color.z,
                    )
                    self.gym.set_rigid_body_color(
                        env,
                        self.object_handles[-1],
                        0,
                        gymapi.MESH_VISUAL_AND_COLLISION,
                        color,
                    )
                elif self.object_type == 4:
                    color = gymapi.Vec3(
                        brick_color.x + np.random.rand() * 0.1,
                        brick_color.y + np.random.rand() * 0.04,
                        brick_color.z + np.random.rand() * 0.05,
                    )
                    self.gym.set_rigid_body_color(
                        env,
                        self.object_handles[-1],
                        0,
                        gymapi.MESH_VISUAL_AND_COLLISION,
                        color,
                    )
                else:
                    self.gym.set_rigid_body_color(
                        env,
                        self.object_handles[-1],
                        0,
                        gymapi.MESH_VISUAL_AND_COLLISION,
                        colors[j % len(colors)],
                    )
        # Process indices
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )
        self.kuka_indices = to_torch(
            self.kuka_indices, dtype=torch.long, device=self.device
        )

    def reset_target(self, env_ids, drawing=False):
        # Random rot
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )
        # Random pos
        rand_x = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            * self.goal_random_range[0]
        )
        rand_y = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            * self.goal_random_range[1]
        )
        rand_z = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            * self.goal_random_range[2]
        )
        self.goal_states[env_ids, 0:3] = self.goal_random_center[
            env_ids
        ] + torch.hstack((rand_x, rand_y, rand_z))

        self.goal_states[env_ids, 3:7] = new_rot

        if drawing:
            # Reset drawing
            self.gym.clear_lines(self.viewer)
            for env_id in env_ids:
                # Draw 6D pose
                # Get pose to cpu
                pos = self.goal_states[env_id, 0:3].to("cpu").numpy()
                rot = self.goal_states[env_id, 3:7].to("cpu").numpy()
                draw_6D_pose(
                    self.gym,
                    self.viewer,
                    self.envs[env_id],
                    pos,
                    rot,
                )
                # Draw bbox
                random_center = self.goal_random_center[env_id].to("cpu").numpy()
                bbox = np.array(
                    [
                        [
                            random_center[0] - self.goal_random_range[0],
                            random_center[1] - self.goal_random_range[1],
                            random_center[2] - self.goal_random_range[2],
                        ],
                        [
                            random_center[0] + self.goal_random_range[0],
                            random_center[1] + self.goal_random_range[1],
                            random_center[2] + self.goal_random_range[2],
                        ],
                    ]
                )
                draw_bbox(self.gym, self.viewer, self.envs[env_id], bbox)

        # Set flag
        self.reset_goal_buf[env_ids] = 0

    def reset_object(self, env_ids):
        # Random rot
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )
        # Random pos
        spawn_height = 0.2
        rand_x = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            * self.goal_random_range[0]
        )
        rand_y = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            * self.goal_random_range[1]
            + spawn_height
        )
        rand_z = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            * self.goal_random_range[2]
        )
        new_pos = self.goal_random_center[env_ids] + torch.hstack(
            (rand_x, rand_y, rand_z)
        )
        self.root_state_tensor[self.object_indices[env_ids], :3] = new_pos
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13]
        )

        object_indices = self.object_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(env_ids),
        )

    def reset_robot(self, env_ids):
        kuka_indices = self.kuka_indices[env_ids].to(torch.int32)

        # Reset robot root
        self.root_state_tensor[self.kuka_indices[env_ids], :3] = torch.zeros(
            3, dtype=torch.float, device=self.device
        )
        y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device)
        init_quat = quat_mul(
            quat_from_angle_axis(np.pi, y_unit_tensor),
            torch.tensor(
                [-0.7071, 0.0000, 0.0000, 0.7071], dtype=torch.float, device=self.device
            ),
        )
        self.root_state_tensor[self.kuka_indices[env_ids], 3:7] = init_quat
        self.root_state_tensor[self.kuka_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.kuka_indices[env_ids], 7:13]
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(kuka_indices),
            len(env_ids),
        )

        # Reset pos control
        kuka_pos = self.kuka_default_dof_pos
        self.kuka_dof_pos[env_ids, :] = kuka_pos
        self.kuka_dof_vel[env_ids, :] = self.kuka_dof_default_vel
        self.prev_targets[env_ids, : self.num_kuka_dofs] = kuka_pos
        self.cur_targets[env_ids, : self.num_kuka_dofs] = kuka_pos

        # Reset state
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(kuka_indices),
            len(env_ids),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(kuka_indices),
            len(env_ids),
        )

        # Reset action
        self.actions = kuka_pos

    def reset_idx(self, env_ids, goal_env_ids):
        # Randomize target poses
        self.reset_target(goal_env_ids, self.debug_viz)
        # Randomize object poses
        self.reset_object(env_ids)
        # Reset robot
        self.reset_robot(env_ids)
        # Set flag
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def reset(self):
        # Reset all envs
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_env_ids, all_env_ids)

    def pre_physics_step(self, actions):
        # Check reset status
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # If only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target(goal_env_ids, self.debug_viz)
        # If goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target(goal_env_ids, self.debug_viz)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.kuka_dof_lower_limits[self.actuated_dof_indices],
            self.kuka_dof_upper_limits[self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.act_moving_average)
            * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
            self.cur_targets[:, self.actuated_dof_indices],
            self.kuka_dof_lower_limits[self.actuated_dof_indices],
            self.kuka_dof_upper_limits[self.actuated_dof_indices],
        )
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[
            :, self.actuated_dof_indices
        ]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pos = self.goal_states[:, :3]
        self.goal_rot = self.goal_states[:, 3:7]

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_grasp_reward(
            self.rew_buf,
            self.reset_buf,
            self.reset_goal_buf,
            self.progress_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.goal_pos,
            self.goal_rot,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.rot_eps,
            self.actions,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.max_consecutive_successes,
            self.av_factor,
        )

        self.extras["consecutive_successes"] = self.consecutive_successes.mean()

        # Print Reward
        print(f"Reward: {self.rew_buf}")

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = (
                self.total_successes + (self.successes * self.reset_buf).sum()
            )

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(
                        self.total_successes / self.total_resets
                    )
                )


@hydra.main(config_name="config_cpu")  # Use cpu for ROS
def hydra_gym_random_app(config):
    env = KukaAllegroGrasp(
        omegaconf_to_dict(config.task),
        config.rl_device,
        config.sim_device,
        config.graphics_device_id,
        config.headless,
        virtual_screen_capture=False,
        force_render=True,
    )
    frame_count = 0
    actions = torch.rand((env.num_envs, env.num_actions)) * 2 - 1

    while True:
        if frame_count % 100 == 0:
            actions = 0.3 * (torch.rand((env.num_envs, env.num_actions)) * 2 - 1)
            env.step(actions)
        else:
            # Repeat the same action if not set
            print(f"Action: {env.actions}")
            env.step(env.actions)

        if frame_count % 300 == 0:
            # Reset the environment
            env.reset()

        frame_count += 1


def get_rlgames_env_creator(  # used to create the vec task
    seed: int,
    task_config: dict,
    task_name: str,
    sim_device: str,
    rl_device: str,
    graphics_device_id: int,
    headless: bool,
    # Used to handle multi-gpu case
    multi_gpu: bool = False,
    post_create_hook: Callable = None,
    virtual_screen_capture: bool = False,
    force_render: bool = False,
):
    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """

        # Create native task and pass custom config
        if task_name == "KukaAllegroGrasp":
            env = KukaAllegroGrasp(
                task_config,
                rl_device,
                sim_device,
                graphics_device_id,
                headless,
                virtual_screen_capture,
                force_render,
            )
        else:
            # Return nothing
            env = None

        if post_create_hook is not None:
            post_create_hook()

        return env

    return create_rlgpu_env


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_grasp_reward(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    max_consecutive_successes: int,
    av_factor: float,
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(
        torch.abs(rot_dist) <= success_tolerance,
        torch.ones_like(reset_goal_buf),
        reset_goal_buf,
    )
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(
            torch.abs(rot_dist) <= success_tolerance,
            torch.zeros_like(progress_buf),
            progress_buf,
        )
        resets = torch.where(
            successes >= max_consecutive_successes, torch.ones_like(resets), resets
        )

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )


if __name__ == "__main__":
    hydra_gym_random_app()