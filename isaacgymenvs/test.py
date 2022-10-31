""" Provide test script for the environment
"""

import isaacgym
import isaacgymenvs

import torch

import numpy as np
import hydra


@hydra.main(config_name="config", config_path="./cfg")
def launch_test_env(cfg):
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
    while True:
        action = env.action_space.sample()
        vec_action = np.repeat(action[np.newaxis, :], cfg.num_envs, axis=0)
        obs, reward, done, info = env.step(torch.from_numpy(vec_action).to(cfg.rl_device))


if __name__ == "__main__":
    launch_test_env()
