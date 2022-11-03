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

```
python train.py task=KukaAllegroGrasp wandb_activate=True wandb_project=KukaAllegroGrasp capture_video=True force_render=False
```

## Trouble shooting

### Video Recording 

Gym 2.6.0 has some bugs in video recording. Use Gym 2.3.0