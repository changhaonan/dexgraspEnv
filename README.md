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
HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python train.py task=KukaAllegroGrasp wandb_activate=True wandb_project=KukaAllegroGrasp force_render=False num_envs=2048
```

Train from checkpoint

```
CUDA_LAUNCH_BLOCKING=1 python train.py task=KukaAllegroGrasp num_envs=6400 max_iterations=1000 wandb_activate=True wandb_project=KukaAllegroGrasp force_render=False checkpoint=runs/KukaAllegroGrasp/nn/last_KukaAllegroGrasp_ep_500_rew_3.3029275.pth
```


## Trouble shooting

### Video Recording 

Gym 2.6.0 has some bugs in video recording. Use Gym 2.3.0