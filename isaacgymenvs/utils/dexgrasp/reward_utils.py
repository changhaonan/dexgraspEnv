import torch
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
