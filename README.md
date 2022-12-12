# dexgraspEnv

DexGrasp env is based on the IsaacGymEnv. But the working target is focusing on the dexterous grasping task.

## Design

Currently, the grasp reward rescribe the distance between object to finger tip. Get finger tips position.

## Todo

- [x] Develop a env debugging environment
- [x] Check the reward function.
	- [x] Use grasp reward.
- [x] Fix the reset structure.
- [x] Check the running.
- [x] Fix the video recording

## Train command

Train from scratch

```
HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python train.py task=AllegroManip wandb_activate=True wandb_project=AllegroManip force_render=False num_envs=8192 max_iterations=5000
```

Train from scratch with video

```
HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python train.py task=AllegroManip capture_video=True capture_video_freq=1500 capture_video_len=100 wandb_activate=True wandb_project=AllegroManip force_render=False num_envs=8192 max_iterations=5000
```

Train from checkpoint

```
HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python train.py task=AllegroManip wandb_activate=True wandb_project=AllegroManip force_render=False num_envs=8192 checkpoint=runs/AllegroManip/nn/AllegroManip.pth max_iterations=4000
```


## Trouble shooting
Gym 2.6.0 has some bugs in video recording. Use Gym 2.3.0

## TODO:

IK: franka_cube_stack.py, Factory_control.py

Check this file: https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf

- We need to decouple the wrist motion and the hand motion.
- So we need three policies here:
	- A policy to control the movement of wrist. (First to train)
	- A policy to control the grasp. (Second to train)
	- A policy to control the release / put down. (Third to train)

This is idea of this week. (Also making the imitation learning pipeline working) Let's train three policy. 

I also want to try some simpler training method (Not rl_games). Easier model makes us easier to change. This week, we just train three seperate policy. And next week, we look for how to combine them. Finally, we make some reports. 