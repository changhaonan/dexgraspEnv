""" Provide tele-operation script for the environment
"""
import isaacgym
import isaacgymenvs
import torch
import numpy as np
import hydra

import math
import cv2
from utils.teleop.mediapipe_hand_pose import MediapipeHandEstimator


@hydra.main(config_name="config", config_path="./cfg")
def launch_test_env(cfg):
    # Fix the num envs to 1
    cfg.num_envs = 1
    # Create the environment
    env = isaacgymenvs.make(
        seed=cfg.seed,
        task=cfg.task,
        num_envs=cfg.num_envs,
        sim_device=cfg.sim_device,
        rl_device=cfg.rl_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        multi_gpu=cfg.multi_gpu,
        virtual_screen_capture=False,
        force_render=cfg.force_render,
    )
    env.reset()

    # Run the environment
    hand_pose_estimator = MediapipeHandEstimator()
    cap = cv2.VideoCapture(0)
    frame = 0
    last_ee_pose = np.zeros(3)
    while True:
        # Capture the hand pose
        ret, frame = cap.read()
        if not ret:
            continue
        joints_3d = hand_pose_estimator.predict_3d_joints(frame)
        # The first 7 dim of action is controlling the wrist
        # The translation motion are all relative motions
        action = env.action_space.sample()
        if len(joints_3d) < 3:
            action[0:3] = last_ee_pose
        else:
            # Rescale x, y to [-1, 1]
            ee_pose = joints_3d[0:3] * 2.0 - 1.0
            # Scale the control
            ee_pose = ee_pose * 1.5
            ee_pose[0] = -ee_pose[0]  # flip the x axis
            ee_pose[1] = -ee_pose[1]  # flip the y axis
            ee_pose[2] = 0.0  # fix the z axis
            action[0:3] = ee_pose
            last_ee_pose = ee_pose
        action[3:7] = 0.0  # fix the rotation
        vec_action = np.repeat(action[np.newaxis, :], cfg.num_envs, axis=0)
        obs, reward, done, info = env.step(
            torch.from_numpy(vec_action).to(cfg.rl_device)
        )
        
        # Update frame
        frame += 1


if __name__ == "__main__":
    launch_test_env()
