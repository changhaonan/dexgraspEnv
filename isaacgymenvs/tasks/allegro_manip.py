import numpy as np
import math
from copy import copy
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.dexgrasp.math_utils import *
from isaacgymenvs.utils.dexgrasp.drawing_utils import draw_6D_pose, draw_3D_pose, draw_bbox, draw_vector
from isaacgymenvs.utils.dexgrasp.reward_utils import compute_pickup_reward, compute_reorient_reward, compute_hold_reward


class AllegroManip(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.manipulate_mode = cfg["env"]["manipulateMode"]
        self.ee_dof_lower_limits = cfg["env"]["eeDofLowerLimits"]
        self.ee_dof_upper_limits = cfg["env"]["eeDofUpperLimits"]
        self.ee_linear_speed_limit = cfg["env"]["eeLinearSpeedLimit"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.hold_still_len = self.cfg["env"]["holdStillLen"]

        self.contact_force_threshold = self.cfg["env"]["contactForceThreshold"]
        self.hold_still_vel_tolerance = self.cfg["env"]["holdStillVelTolerance"]
        self.angvel_scale = self.cfg["env"]["angVelScale"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.contact_force_reward_scale = self.cfg["env"]["contactForceRewardScale"]
        self.hold_still_reward_scale = self.cfg["env"]["holdStillRewardScale"]

        self.goal_trans_tolerance = self.cfg["env"]["goalTransTolerance"]
        self.goal_rot_tolerance = self.cfg["env"]["goalRotTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.slide_penalty = self.cfg["env"]["slidePenalty"]
        self.max_dist_slide = self.cfg["env"]["maxDistSlide"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.force_range = self.cfg["env"]["forceRange"]

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)
        self.transition_scale = self.cfg["env"]["transitionScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen", "can", "banana", "mug", "brick"]
        self.ignore_z = (self.object_type == "pen")
        self.asset_files_dict = {}
        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock")
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg")
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen")
            self.asset_files_dict["can"] = self.cfg["env"]["asset"].get("assetFileNameCan")
            self.asset_files_dict["banana"] = self.cfg["env"]["asset"].get("assetFileNameBanana")
            self.asset_files_dict["mug"] = self.cfg["env"]["asset"].get("assetFileNameMug")
            self.asset_files_dict["brick"] = self.cfg["env"]["asset"].get("assetFileNameBrick")
        
        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        # domain ramdomization
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "full_no_vel": 57,
            "full": 79,
            "full_state": 98
        }

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 88

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 16 + 7

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # contact_tensor
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(contact_tensor).view(-1, 3)

        if self.obs_type == "full_state" or self.asymmetric_obs:
        #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # base control tensors
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # allocate buffer
        self.goal_dist_buf = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self.rot_dist_buf = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self.hand_contact_force = torch.zeros((self.num_envs, self.num_hand_part), dtype=torch.float, device=self.device)
        self.contact_force_sum_buf = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self.hold_still_count_buf = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self.reach_goal_buf = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # for vis
        self.ee_attr_pos = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.ee_attr_shift = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        # set camera
        cam_pos = gymapi.Vec3(0.0, -0.3, 1.5)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # if randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        shadow_hand_asset_file = "urdf/kuka_allegro_description/allegro.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]
        
        # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.armature = 0.001
        table_asset_options.fix_base_link = True
        table_asset_options.thickness = 0.002
        table_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        table_dims = gymapi.Vec3(1.0, 0.6, 0.4)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options
        )

        # load shadow hand_ asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = False
        hand_asset_options.collapse_fixed_joints = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 100  # 0.01
        hand_asset_options.linear_damping = 100  # 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            hand_asset_options.use_physx_armature = True
        hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, hand_asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        print("Num dofs: ", self.num_shadow_hand_dofs)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs #self.gym.get_asset_actuator_count(shadow_hand_asset)

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

            # print("Max effort: ", shadow_hand_dof_props['effort'][i])
            shadow_hand_dof_props['effort'][i] = 0.5
            shadow_hand_dof_props['stiffness'][i] = 3
            shadow_hand_dof_props['damping'][i] = 0.1
            shadow_hand_dof_props['friction'][i] = 0.01
            shadow_hand_dof_props['armature'][i] = 0.001
        print("lower: ", self.shadow_hand_dof_lower_limits)
        print("upper: ", self.shadow_hand_dof_upper_limits)
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)
        self.ee_dof_lower_limits = to_torch(self.ee_dof_lower_limits, device=self.device)
        self.ee_dof_upper_limits = to_torch(self.ee_dof_upper_limits, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.57, self.up_axis_idx))
        shadow_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -1.0 * np.pi) * \
            gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * np.pi) * \
            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.25 * np.pi)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = shadow_hand_start_pose.p.x
        pose_dy, pose_dz = -0.12, -0.10

        object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = shadow_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + 2
        max_agg_shapes = self.num_shadow_hand_shapes + 2

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.hand_part_indices = []

        self.object_rb_indices = []
        
        self.ee_attr_handles = []
        self.ee_attr_base_poses = []
        #self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(shadow_hand_rb_count, shadow_hand_rb_count + object_rb_count))

        # creating force sensor
        body_names = [self.gym.get_asset_rigid_body_name(shadow_hand_asset, i) for i in range(shadow_hand_rb_count)]
        body_indices = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in body_names]
        
        # sensor Properties
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = True
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = True

        sensor_pose = gymapi.Transform()
        for body_idx in body_indices:
            self.gym.create_asset_force_sensor(shadow_hand_asset, body_idx, sensor_pose, sensor_props)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add table
            self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # bind attractor to ee
            ee_name = "allegro_mount"
            body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, shadow_hand_actor)
            body_props = self.gym.get_actor_rigid_body_states(env_ptr, shadow_hand_actor, gymapi.STATE_POS)
            attractor_properties = gymapi.AttractorProperties()
            attractor_properties.stiffness = 1e6
            attractor_properties.damping = 1e5 # 5e2
            ee_handle = self.gym.find_actor_rigid_body_handle(
                env_ptr, shadow_hand_actor, ee_name)
            attractor_properties.target = body_props["pose"][:][body_dict[ee_name]]

            # by default, offset pose is set to origin, so no need to set it
            # set all direction attraction
            attractor_properties.axes = gymapi.AXIS_ALL

            # attractor_properties.target.p.z=0.1
            attractor_properties.rigid_handle = ee_handle
            attractor_handle = self.gym.create_rigid_body_attractor(env_ptr, attractor_properties)

            self.ee_attr_handles.append(attractor_handle)
            self.ee_attr_base_poses.append(attractor_properties.target)
                
            # create fingertip force-torque sensors
            # if self.obs_type == "full_state" or self.asymmetric_obs:
            #     for ft_handle in self.fingertip_handles:
            #         env_sensors = []
            #         env_sensors.append(self.gym.create_force_sensor(env_ptr, ft_handle, sensor_pose))
            #         self.sensors.append(env_sensors)

            #     self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)
            
            # add finger all parts indices
            self.num_hand_part = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
            hand_part_dict = self.gym.get_actor_rigid_body_dict(env_ptr, shadow_hand_actor)
            for rel_idx in hand_part_dict.values():
                part_sim_idx = self.gym.get_actor_rigid_body_index(env_ptr, shadow_hand_actor, rel_idx, gymapi.DOMAIN_SIM)
                self.hand_part_indices.append(part_sim_idx)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_rb_indices.append(self.gym.get_actor_rigid_body_index(env_ptr, object_handle, 0, gymapi.DOMAIN_SIM))
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        # goal initialization
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] += 0.04  # goal pos is higher than init
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.goal_force = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

        # self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.object_rb_indices = to_torch(self.object_rb_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

        self.hand_part_indices = to_torch(self.hand_part_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        if self.manipulate_mode == "pickup":
            self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], self.successes[:], self.goal_dist_buf = compute_pickup_reward(
                self.rew_buf, self.reset_buf, self.progress_buf, self.successes,
                self.max_episode_length, self.object_pos, self.goal_pos,
                self.dist_reward_scale, self.actions, self.action_penalty_scale,
                self.goal_trans_tolerance, self.reach_goal_bonus, 
                self.max_dist_slide, self.slide_penalty
            )
        elif self.manipulate_mode == "hold":
            self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], self.successes[:], self.hold_still_count_buf[:], self.goal_dist_buf, self.contact_force_sum_buf[:], self.reach_goal_buf[:] = compute_hold_reward(
                self.rew_buf, self.reset_buf, self.progress_buf, self.successes,
                self.max_episode_length, 
                self.object_pos, self.object_linvel, self.object_angvel, self.angvel_scale,
                self.goal_pos, self.dist_reward_scale, 
                self.hand_contact_force, self.contact_force_threshold, self.contact_force_reward_scale,
                self.hold_still_count_buf, self.hold_still_len, self.hold_still_reward_scale, self.hold_still_vel_tolerance,
                self.actions, self.action_penalty_scale,
                self.goal_trans_tolerance, self.reach_goal_bonus, 
                self.max_dist_slide, self.slide_penalty
            )
        elif self.manipulate_mode == "reorient":
            self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:], self.goal_dist_buf[:], self.rot_dist_buf[:] = compute_reorient_reward(
                self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
                self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
                self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
                self.goal_rot_tolerance, self.reach_goal_bonus, self.max_dist_slide, self.slide_penalty,
                self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
            )

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        # self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.hand_contact_force = self.contact_forces[self.hand_part_indices].reshape(self.num_envs, -1, 3) 

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
             self.compute_full_state()
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)

            self.obs_buf[:, 16:23] = self.object_pose
            self.obs_buf[:, 23:30] = self.goal_pose
            self.obs_buf[:, 30:34] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # 3*self.num_fingertips = 15
            #self.obs_buf[:, 42:57] = self.fingertip_pos.reshape(self.num_envs, 15)

            self.obs_buf[:, 34:50] = self.actions
        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel

            # 2*16 = 32 -16
            self.obs_buf[:, 32:39] = self.object_pose
            self.obs_buf[:, 39:42] = self.object_linvel
            self.obs_buf[:, 42:45] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 45:52] = self.goal_pose
            self.obs_buf[:, 52:56] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # 13*self.num_fingertips = 65 4*13 = 52
            # self.obs_buf[:, 72:137] = self.fingertip_state.reshape(self.num_envs, 65)

            self.obs_buf[:, 56:72] = self.actions

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                      self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.states_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            self.states_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_shadow_hand_dofs  # 48
            self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # fingertip observations, state(pose and vel) + force-torque sensors
            # todo - add later
            # num_ft_states = 13 * self.num_fingertips  # 65
            # num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 72
            # self.states_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            # self.states_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
            #                 num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor
            # obs_end = 96 + 65 + 30 = 191
            obs_end = fingertip_obs_start #+ num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end:obs_end + self.num_actions] = self.actions

            # goal force
            goal_force_start = obs_end + self.num_actions
            # obs_total = goal_force_start + 3 = 88 + 3 = 91
            self.obs_buf[:, goal_force_start:goal_force_start + 3] = self.goal_force
        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                      self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_shadow_hand_dofs  # 48
            self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # fingertip observations, state(pose and vel) + force-torque sensors
            # todo - add later
            # num_ft_states = 13 * self.num_fingertips  # 65
            # num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 72
            # self.states_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            # self.states_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
            #                 num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor

            # obs_end = 96 + 65 + 30 = 191
            obs_end = fingertip_obs_start #+ num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end:obs_end + self.num_actions] = self.actions

            # goal force
            goal_force_start = obs_end + self.num_actions
            # obs_total = goal_force_start + 3 = 88 + 3 = 91
            self.obs_buf[:, goal_force_start:goal_force_start + 3] = self.goal_force

    def reset_target(self, env_ids, apply_reset=False):
        # reset the target pose
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        
        # reset the target force
        self.goal_force[env_ids] = torch_rand_float(self.force_range[0], self.force_range[1], (len(env_ids), 3), device=self.device)

        # reset the goal buf
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        """ control the hand base and hand joint: (dim: 23 = 7 + 16)
            - actions[0:3] controls the translation of hand base
            - actions[3:7] controls the rotation of hand base
            - actions[7:] controls the joint angles of the hand
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 7:],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            if (self.device == "cpu"):
                # apply shift to attractor, hand mount
                self.apply_attractor_shift(self.actions[:, 0:7]) 

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.prev_targets))

        # apply random force to manipulated object
        force_after_reach = torch.where(self.reach_goal_buf[:, None].expand(-1, 3) == 1, self.goal_force, torch.zeros_like(self.goal_force))
        self.rb_forces.view(-1, 3)[self.object_rb_indices, :] = force_after_reach
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        self.log_metric()

        if self.viewer and self.debug_viz:
            self.draw_auxilary()

    def apply_attractor_shift(self, ee_actions):
        # scale the offset
        ee_actions = scale(ee_actions, self.ee_dof_lower_limits, self.ee_dof_upper_limits)
        ee_linear_speeds = ee_actions[:, 0:3] - self.ee_attr_shift[:, 0:3]
        ee_linear_speeds_norm = torch.norm(ee_linear_speeds, dim=1, keepdim=True)
        ee_linear_speeds = torch.where(
            ee_linear_speeds_norm > self.ee_linear_speed_limit,
            self.ee_linear_speed_limit * ee_linear_speeds / ee_linear_speeds_norm,
            ee_linear_speeds
        )
        self.ee_attr_shift[:, 0:3] += ee_linear_speeds
        # average shift
        av_factor = 0.5
        self.ee_attr_shift[:, 3:7] = (1 - av_factor) * self.ee_attr_shift[:, 3:7] + av_factor * ee_actions[:, 3:7]
        # clamp the shift
        self.ee_attr_shift = tensor_clamp(self.ee_attr_shift, self.ee_dof_lower_limits, self.ee_dof_upper_limits)

        for idx_env in range(self.num_envs):
            # apply translation shift
            attr_pose = copy(self.ee_attr_base_poses[idx_env])
            attr_pose.p.x += self.ee_attr_shift[idx_env, 0]
            attr_pose.p.y += self.ee_attr_shift[idx_env, 1]
            attr_pose.p.z += self.ee_attr_shift[idx_env, 2]

            # apply rotation shift
            # attr_rot_offset = gymapi.Quat(
            #     self.ee_attr_shift[idx_env, 3], 
            #     self.ee_attr_shift[idx_env, 4],
            #     self.ee_attr_shift[idx_env, 5],
            #     self.ee_attr_shift[idx_env, 6])
            # attr_pose.r = quaternion_mul(
            #     attr_pose.r, attr_rot_offset.normalize())  # Right multiply

            self.gym.set_attractor_target(
                self.envs[idx_env], self.ee_attr_handles[idx_env], attr_pose)
            
            # Save for vis
            if self.viewer and self.debug_viz:
                # save prev pos
                self.ee_attr_pos[idx_env, 0] = attr_pose.p.x
                self.ee_attr_pos[idx_env, 1] = attr_pose.p.y
                self.ee_attr_pos[idx_env, 2] = attr_pose.p.z
                self.ee_attr_pos[idx_env, 3] = attr_pose.r.x
                self.ee_attr_pos[idx_env, 4] = attr_pose.r.y
                self.ee_attr_pos[idx_env, 5] = attr_pose.r.z
                self.ee_attr_pos[idx_env, 6] = attr_pose.r.w

    def draw_auxilary(self):
        # draw axes on target object
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        for i in range(self.num_envs):
            # draw attractor
            # draw_6D_pose(self.gym, self.viewer, self.envs[i], 
            #     self.ee_attr_pos[i, 0:3], 
            #     self.ee_attr_pos[i, 3:7])
            
            # draw out the target force, centered at the object
            object_center = self.object_pos[i].cpu().numpy()
            force = self.goal_force[i].cpu().numpy() * 0.1  # scale down
            draw_vector(self.gym, self.viewer, self.envs[i], object_center, force, color=(1, 1, 0))

    def log_metric(self):
        for idx in range(min(self.num_envs, 4)):
            if self.manipulate_mode == "pickup":
                print(f"Env {idx}, prog: {self.progress_buf[idx]}, reward: {self.rew_buf[idx]}, goal_dist: {self.goal_dist_buf[idx]}, reset: {self.reset_buf[idx]}.")
            elif self.manipulate_mode == "hold":
                print(f"Env {idx}, prog: {self.progress_buf[idx]}, reward: {self.rew_buf[idx]}, goal_dist: {self.goal_dist_buf[idx]}, reset: {self.reset_buf[idx]}, contact_sum: {self.contact_force_sum_buf[idx]}, hold still: {self.hold_still_count_buf[idx]}, reach: {self.reach_goal_buf[idx]}.")
            elif self.manipulate_mode == "reorient":
                print(f"Env {idx}, prog: {self.progress_buf[idx]}, reward: {self.rew_buf[idx]}, goal_dist: {self.goal_dist_buf[idx]}, rot_dist: {self.rot_dist_buf[idx]}, reset: {self.reset_buf[idx]}.")


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
