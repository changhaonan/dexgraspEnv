""" Provide tele-operation script for the environment
"""
import isaacgym
import isaacgymenvs
import torch
import numpy as np
import hydra

import math


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

    frame = 0
    while True:
        # The first 7 dim of action is controlling the wrist
        # The translation motion are all relative motions
        action = env.action_space.sample()
        action[0] = math.cos(frame / 100.0)  # x-motion
        action[1] = math.sin(frame / 100.0)  # y-motion
        action[2] = math.sin(frame / 100.0)  # z-motion
        action[3:7] = 0.0  # fix the rotation
        vec_action = np.repeat(action[np.newaxis, :], cfg.num_envs, axis=0)
        obs, reward, done, info = env.step(
            torch.from_numpy(vec_action).to(cfg.rl_device)
        )
        
        # Update frame
        frame += 1


if __name__ == "__main__":
    launch_test_env()
