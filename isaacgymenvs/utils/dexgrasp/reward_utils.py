import torch
from isaacgym.torch_utils import *
import roma


@torch.jit.script
def compute_grasp_reward_v2(
    rew_buf,
    reset_buf,
    progress_buf,
    successes,
    max_episode_length: int,
    object_pos,
    goal_pos,
    hand_joint_pos,
    fingertip_pos,
    actions,
    dist_object_palm,
    dist_grasp_palm_tol: float,
    dist_grasp_finger_tol: float,
    dist_goal_tol: float,
    reach_goal_bonus: float,
    coef_palm: float,
    coef_goal: float,
    coef_hand_open_penalty: float,
    coef_action_penalty: float,
    coef_finger_contact: float,
    av_factor,
):
    # Update grasp phase: [0: approach, 1: grasp]
    phase_grasp = torch.where(
        dist_object_palm < dist_grasp_palm_tol,
        torch.ones_like(dist_object_palm),
        torch.zeros_like(dist_object_palm),
    )

    # 1. Phase zero: approach the object
    # r_palm: Distance from the palm to the object, a constant reward
    r_palm = (
        dist_grasp_palm_tol / torch.clamp(dist_object_palm, min=dist_grasp_palm_tol)
        - 1.0
    ) * coef_palm
    # r_palm = torch.where(phase_grasp == 1, torch.zeros_like(r_palm), r_palm)

    # r_hand_open: Make hand joint close to 0. This is a penalty
    r_hand_open = torch.norm(hand_joint_pos, p=2, dim=-1) * coef_hand_open_penalty
    r_hand_open = torch.where(
        phase_grasp == 1, torch.zeros_like(r_hand_open), r_hand_open
    )

    # 2. Phase one: move the object to the goal
    # r_goal: Distance from the object to the goal
    dist_goal = torch.norm(object_pos - goal_pos, p=2, dim=-1)
    r_goal = (dist_goal_tol / torch.clamp(dist_goal, min=dist_goal_tol)) * coef_goal
    r_goal = torch.where(phase_grasp == 0, torch.zeros_like(r_goal), r_goal)

    # r_finger_contact: Make the fingers close to the object
    # This term is always positive, serverd as a bonus
    dist_finger_contact = torch.norm(
        fingertip_pos - object_pos.repeat(1, 4).reshape(-1, 3), p=2, dim=-1
    ).reshape(-1, 4)
    dist_finger_contact[:, 0] = dist_finger_contact[:, 0] * 2.0  # Amplify the thumb dist

    r_finger_contact = (
        dist_grasp_finger_tol
        / torch.clamp(dist_finger_contact.mean(dim=-1), min=dist_grasp_finger_tol)
        * coef_finger_contact
    )
    r_finger_contact = torch.where(
        phase_grasp == 0, torch.zeros_like(r_finger_contact), r_finger_contact
    )

    # Total reward
    reward = r_palm + r_hand_open + r_goal + r_finger_contact

    # Sucess resets
    success_resets = torch.where(
        torch.abs(dist_goal) <= dist_goal_tol,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    successes = successes + success_resets

    # Reach bonus: object-plam is close enough to the object
    reward = torch.where(success_resets == 1, reward + reach_goal_bonus, reward)

    # Timeout resets
    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(success_resets), success_resets)

    return (
        reward,
        resets,
        success_resets,
        progress_buf,
        successes,
        phase_grasp,
        r_palm,
        r_hand_open,
        r_goal,
        r_finger_contact,
    )


@torch.jit.script
def compute_pickup_reward(
    rew_buf, reset_buf, progress_buf, successes,
    max_episode_length: float, object_pos, target_pos,
    dist_reward_scale: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float,
    max_dist_slide: float, slide_penalty: float
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    dist_rew = success_tolerance/torch.clamp(goal_dist, min=success_tolerance) * dist_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count and resets
    reach_goal = torch.where(goal_dist <= success_tolerance, torch.ones_like(reset_buf), reset_buf)
    successes = successes + reach_goal
    resets = torch.where(reach_goal == 1, torch.ones_like(reset_buf), reset_buf)

    # Success bonus: reach the
    reward = torch.where(reach_goal == 1, reward + reach_goal_bonus, reward)
    
    # Check if the object slide from hand too far
    resets = torch.where(goal_dist > max_dist_slide, torch.ones_like(resets), resets)
    reward = torch.where(goal_dist > max_dist_slide, reward + slide_penalty, reward)

    # Timeout resets
    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    return reward, resets, progress_buf, successes, goal_dist


@torch.jit.script
def compute_hold_reward(
    rew_buf, reset_buf, progress_buf, successes,
    max_episode_length: float, 
    object_pos, object_linvel, object_angvel, angvel_scale: float,
    target_pos, dist_reward_scale: float,
    hand_contact_force, contact_force_threshold: float, contact_reward_scale: float,
    hold_still_count_buf, hold_still_len, hold_still_reward_scale: float, hold_still_vel_threshold: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, success_bonus: float,
    max_dist_slide: float, slide_penalty: float
):
    # Configuration
    num_envs = rew_buf.shape[0]

    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    dist_rew = success_tolerance/torch.clamp(goal_dist, min=success_tolerance) * dist_reward_scale

    # Action penalty
    action_penalty = torch.sum(actions ** 2, dim=-1)
    
    # Find out which envs hit the goal and progress the hold still count
    reach_goal = torch.where(goal_dist <= success_tolerance, torch.ones_like(reset_buf), reset_buf)
    hold_still_count_buf = torch.where(reach_goal == 1, hold_still_count_buf + 1, hold_still_count_buf)

    # Contact force reward
    contact_force_rigid_body = torch.norm(hand_contact_force, p=2, dim=-1)
    contact_existence = torch.where(contact_force_rigid_body > contact_force_threshold, torch.ones_like(contact_force_rigid_body), torch.zeros_like(contact_force_rigid_body))
    contact_existence_sum = torch.sum(contact_existence, dim=-1)
    contact_rew = 1.0 / torch.clamp(contact_existence_sum, min=1.0)

    # Hold still reward (Compute when the object is close enough to the target)
    object_vel_norm = torch.norm(object_linvel, p=2, dim=-1) + torch.norm(object_angvel, p=2, dim=-1) * angvel_scale
    hold_still_rew = 1.0 / torch.clamp(object_vel_norm, min=hold_still_vel_threshold) 
    hold_still_rew = torch.where(reach_goal == 1, hold_still_rew, torch.zeros_like(hold_still_rew))
    hold_still_success = torch.where(hold_still_count_buf >= hold_still_len, torch.ones_like(reset_buf), reset_buf)
    hold_still_count_buf = torch.where(hold_still_success == 1, torch.zeros_like(hold_still_count_buf), hold_still_count_buf)
    
    # Success when holding still for a time period
    successes = successes + hold_still_success
    resets = torch.where(hold_still_success == 1, torch.ones_like(reset_buf), reset_buf)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + action_penalty * action_penalty_scale + contact_rew * contact_reward_scale + hold_still_rew * hold_still_reward_scale

    # Success bonus: reach the goal
    reward = torch.where(hold_still_success == 1, reward + success_bonus, reward)
    
    # Check if the object slide from hand too far
    resets = torch.where(goal_dist > max_dist_slide, torch.ones_like(resets), resets)
    reward = torch.where(goal_dist > max_dist_slide, reward + slide_penalty, reward)

    # Timeout resets
    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    return reward, resets, progress_buf, successes, hold_still_count_buf, goal_dist, contact_existence_sum, reach_goal


# @torch.jit.script
def compute_reorient_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, max_dist_slide: float,
    slide_penalty: float, max_consecutive_successes: int, av_factor, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Slide penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= max_dist_slide, reward + slide_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= max_dist_slide, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward + 0.5 * slide_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes, goal_dist, rot_dist