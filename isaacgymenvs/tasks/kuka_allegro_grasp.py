from copy import copy
import math
import os

import numpy as np
import matplotlib.pyplot as plt
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

import torch
import roma
import wandb

from isaacgymenvs.utils.dexgrasp.math_utils import quaternion_mul
from isaacgymenvs.utils.dexgrasp.drawing_utils import (
    draw_6D_pose,
    draw_3D_pose,
    draw_bbox,
)
from isaacgymenvs.utils.dexgrasp.reward_utils import compute_grasp_reward_v2


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
        self.dist_grasp_palm_tol = self.cfg["env"]["distGraspPalmTolerance"]
        self.dist_grasp_finger_tol = self.cfg["env"]["distGraspFingerTolerance"]
        self.dist_goal_tol = self.cfg["env"]["distGoalTolerance"]

        self.coef_palm = self.cfg["env"]["coefPalm"]
        self.coef_goal = self.cfg["env"]["coefGoal"]
        self.coef_hand_open_penalty = self.cfg["env"]["coefHandOpenPenalty"]
        self.coef_action_penalty = self.cfg["env"]["coefActionPenalty"]
        self.coef_finger_contact = self.cfg["env"]["coefFingerContact"]

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]

        self.vel_obs_scale = 0.2  # Scale factor of velocity based observations
        self.force_torque_obs_scale = (
            10.0  # Scale factor of velocity based observations
        )

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
        self.cfg["env"]["numObservations"] = 89
        self.cfg["env"]["numStates"] = 42
        self.cfg["env"]["numActions"] = 23

        # Video setting
        self.render_mode = "rgb_array"

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

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(-1, 13)
        self.num_bodies = self.rigid_body_states.shape[0] / self.num_envs

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

        # Allocate buffer
        self.r_palm_buf = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )
        self.r_hand_open_buf = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )
        self.r_goal_buf = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )
        self.r_finger_contact_buf = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )
        self.phase_grasp_buf = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )

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
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )

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

        self.spawn_height = 0.0

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

        # Fingertips config
        self.fingertips_handles = []
        self.fingertips_indices = []
        fingertips_links = [
            "thumb_link_3",
            "index_link_3",
            "middle_link_3",
            "ring_link_3",
        ]
        self.num_fingertips = len(fingertips_links)

        # Palm config
        self.palm_indices = []
        self.palm_pos_offset = torch.tensor(
            [0.0, 0.0, 0.15], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        palm_rot_vec_offset = torch.tensor(
            [0.0, 0.0, 2.37], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self.palm_rot_offset = roma.rotvec_to_unitquat(palm_rot_vec_offset)

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

            # Finger tips
            for link_name in fingertips_links:
                fingertip_handle = self.gym.find_actor_rigid_body_handle(
                    env, kuka_handle, link_name
                )
                fingertip_indice = self.gym.find_actor_rigid_body_index(
                    env, kuka_handle, link_name, gymapi.DOMAIN_SIM
                )
                self.fingertips_handles.append(fingertip_handle)
                self.fingertips_indices.append(fingertip_indice)

            # Palm : Offset from wrist
            palm_indice = self.gym.find_actor_rigid_body_index(
                env,
                kuka_handle,
                "iiwa7_link_7",
                gymapi.DOMAIN_SIM,
            )
            self.palm_indices.append(palm_indice)

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

                object_pose.p = gymapi.Vec3(x, y + self.spawn_height, z)

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
        self.fingertips_indices = to_torch(
            self.fingertips_indices, dtype=torch.long, device=self.device
        )
        self.palm_indices = to_torch(
            self.palm_indices, dtype=torch.long, device=self.device
        )

    def reset_target(self, env_ids):
        target_y_offset = 0.3

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
        ) + target_y_offset
        rand_z = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            * self.goal_random_range[2]
        )
        self.goal_states[env_ids, 0:3] = self.goal_random_center[
            env_ids
        ] + torch.hstack((rand_x, rand_y, rand_z))

        self.goal_states[env_ids, 3:7] = new_rot

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
        rand_x = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            * self.goal_random_range[0]
        )
        rand_y = (
            torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            * self.goal_random_range[1]
            + self.spawn_height
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
        pi_device = torch.Tensor([np.pi]).float().to(self.device)
        init_quat = quat_mul(
            quat_from_angle_axis(pi_device, y_unit_tensor),
            torch.tensor(
                [[-0.7071, 0.0000, 0.0000, 0.7071]],
                dtype=torch.float,
                device=self.device,
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
        self.reset_target(goal_env_ids)
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

        # Compute observations
        self.obs_dict["obs"] = torch.clamp(
            self.obs_buf, -self.clip_obs, self.clip_obs
        ).to(self.rl_device)
        # Asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def pre_physics_step(self, actions):
        # Check reset status
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # If only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target(goal_env_ids)
        # If goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target(goal_env_ids)

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
        self.compute_reward()  # V2 is grasp, V1 is pickup
        if self.debug_viz:
            self.draw_auxiliary()

    def compute_observations(self):
        # Compute state
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Object-observation
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        if hasattr(self, "object_pos"):
            self.object_prev_pos = self.object_pos
        else:
            self.object_prev_pos = torch.zeros(
                [self.num_objects, 3], dtype=torch.float32, device=self.rl_device
            )
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pos = self.goal_states[:, :3]
        self.goal_rot = self.goal_states[:, 3:7]

        # Hand joint position
        self.hand_joint_pos = self.kuka_dof_pos[:, 7:23]

        # Fingertip-observation
        self.fingertip_pos = self.rigid_body_states[self.fingertips_indices, 0:3]

        # Palm-observation
        self.wrist_pos = self.rigid_body_states[self.palm_indices, 0:3]
        self.wrist_rot = self.rigid_body_states[self.palm_indices, 3:7]
        self.palm_pos = (
            roma.quat_action(self.wrist_rot, self.palm_pos_offset) + self.wrist_pos
        )
        self.palm_rot = roma.quat_product(self.wrist_rot, self.palm_rot_offset)

        # Palm reaching reward (Consider orientation)
        object_palm_diff = self.object_pos - self.palm_pos
        object_palm_diff_in_palm = roma.quat_action(
            roma.quat_inverse(self.palm_rot), object_palm_diff
        )
        object_palm_diff_in_palm[:, 1] = torch.where(
            object_palm_diff_in_palm[:, 1] < 0,
            torch.tensor(10.0, dtype=torch.float32, device=self.device),
            object_palm_diff_in_palm[:, 1],
        )
        self.dist_object_palm = torch.norm(object_palm_diff_in_palm, dim=-1)
        # Compute full observation
        self.compute_full_observations()

    def compute_full_observations(self):
        # Kuka: pos & vel
        self.obs_buf[:, 0 : self.num_kuka_dofs] = unscale(
            self.kuka_dof_pos, self.kuka_dof_lower_limits, self.kuka_dof_upper_limits
        )
        self.obs_buf[:, self.num_kuka_dofs : 2 * self.num_kuka_dofs] = (
            self.vel_obs_scale * self.kuka_dof_vel
        )
        # Object: pos & vel
        self.obs_buf[
            :, 2 * self.num_kuka_dofs : 2 * self.num_kuka_dofs + 7
        ] = self.object_pose
        self.obs_buf[
            :, 2 * self.num_kuka_dofs + 7 : 2 * self.num_kuka_dofs + 10
        ] = self.object_linvel
        self.obs_buf[:, 2 * self.num_kuka_dofs + 10 : 2 * self.num_kuka_dofs + 13] = (
            self.vel_obs_scale * self.object_angvel
        )
        # Target: pos
        self.obs_buf[
            :, 2 * self.num_kuka_dofs + 13 : 2 * self.num_kuka_dofs + 20
        ] = self.goal_states[:, :7]
        # Action
        self.obs_buf[
            :, 2 * self.num_kuka_dofs + 20 : 3 * self.num_kuka_dofs + 20
        ] = self.actions

    def draw_auxiliary(self):
        self.gym.clear_lines(self.viewer)

        # Draw random
        for env_idx in range(self.num_envs):
            # Draw 6D pose
            # Get pose to cpu
            pos = self.goal_states[env_idx, 0:3].to("cpu").numpy()
            rot = self.goal_states[env_idx, 3:7].to("cpu").numpy()
            draw_6D_pose(
                self.gym,
                self.viewer,
                self.envs[env_idx],
                pos,
                rot,
            )
            # Draw bbox
            random_center = self.goal_random_center[env_idx].to("cpu").numpy()
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
            draw_bbox(self.gym, self.viewer, self.envs[env_idx], bbox)

        # Draw observation
        for env_idx in range(self.num_envs):
            # Draw finger
            for finger_idx in range(self.num_fingertips):
                draw_3D_pose(
                    self.gym,
                    self.viewer,
                    self.envs[env_idx],
                    self.fingertip_pos[env_idx * self.num_fingertips + finger_idx],
                )
            # Draw object
            draw_3D_pose(
                self.gym,
                self.viewer,
                self.envs[env_idx],
                self.object_pos[env_idx],
                sphere_radius=0.1,
                color=(0, 0, 1),
            )

        # Draw link line
        for env_idx in range(self.num_envs):
            # We fix mid point to be middle finger's middle joint
            mid_point_index = self.gym.find_actor_rigid_body_index(
                self.envs[env_idx],
                self.kuka_handles[env_idx],
                "middle_link_0",
                gymapi.DOMAIN_ENV,
            )

            # 3d coordinates of object center and kuka hand center
            p1_x, p1_y, p1_z = self.object_pos[env_idx]
            p2_x, p2_y, p2_z = self.rigid_body_states[
                int(env_idx * self.num_bodies + mid_point_index), :3
            ]

            # Convert coordiates to 3d gymapi vectors
            object_center = gymapi.Vec3(p1_x, p1_y, p1_z)
            kuka_hand_center = gymapi.Vec3(p2_x, p2_y, p2_z)

            # Calculate manhattan distance between object and allegro hand
            obj_hand_distance = math.sqrt(
                (object_center.x - kuka_hand_center.x) ** 2
                + (object_center.y - kuka_hand_center.y) ** 2
                + (object_center.z - kuka_hand_center.z) ** 2
            )

            # Choose color based on distance between object and robotic hand
            colors = plt.cm.cool(obj_hand_distance)
            color = gymapi.Vec3(colors[0], colors[1], colors[2])

            # Draw line betwee object and hand for tracking
            gymutil.draw_line(
                object_center,
                kuka_hand_center,
                color,
                self.gym,
                self.viewer,
                self.envs[env_idx],
            )

        # Draw palm coordinates
        for env_idx in range(self.num_envs):
            draw_6D_pose(
                self.gym,
                self.viewer,
                self.envs[env_idx],
                self.palm_pos[env_idx].to("cpu").numpy(),
                self.palm_rot[env_idx].to("cpu").numpy(),
                axis_length=0.3,
                color=(0, 1, 0),
            )

    def compute_reward(self):
        object_pos_rep = self.object_pos.repeat(1, self.num_fingertips).view(
            self.num_envs, -1
        )
        fingertip_pos_view = self.fingertip_pos.view(self.num_envs, -1)
        # Compute reward
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.phase_grasp_buf[:],
            self.r_palm_buf[:],
            self.r_hand_open_buf[:],
            self.r_goal_buf[:],
            self.r_finger_contact_buf[:],
        ) = compute_grasp_reward_v2(
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
            self.successes,
            self.max_episode_length,
            self.object_pos,
            self.goal_pos,
            self.hand_joint_pos,
            self.fingertip_pos,
            self.actions,
            self.dist_object_palm,
            self.dist_grasp_palm_tol,
            self.dist_grasp_finger_tol,
            self.dist_goal_tol,
            self.reach_goal_bonus,
            self.coef_palm,
            self.coef_goal,
            self.coef_hand_open_penalty,
            self.coef_action_penalty,
            self.coef_finger_contact,
            self.av_factor,
        )

        self.extras["consecutive_successes"] = self.consecutive_successes.mean()

        # Print stats
        self.log_wandb()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            self.total_successes = self.total_successes + self.successes.sum()

            if self.total_resets > 0 and self.total_successes > 0:
                print(
                    "Average Successes = {:.1f}".format(
                        self.total_successes / self.total_resets
                    )
                )

    def log_wandb(self):
        print("=============== ENV STATS ===============")
        for env_idx in range(min(4, self.num_envs)):
            print(
                f"Env {env_idx}:",
                f"rew = {self.rew_buf[env_idx]},",
                f"r_palm = {self.r_palm_buf[env_idx]},",
                f"r_hand_open = {self.r_hand_open_buf[env_idx]},",
                f"r_goal = {self.r_goal_buf[env_idx]},",
                f"r_finger_contact = {self.r_finger_contact_buf[env_idx]},",
                f"prog = {self.progress_buf[env_idx]},",
                f"phase = {self.phase_grasp_buf[env_idx]},",
                f"succ = {self.successes[env_idx]},",
                f"reset = {self.reset_buf[env_idx]}.",
            )

        # Log stats to wandb
        # wandb.log()

# JIT Script
@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )


if __name__ == "__main__":
    hydra_gym_random_app()
