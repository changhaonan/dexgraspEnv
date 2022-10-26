""" Provide math utilities for isaacgym. """
from isaacgym import gymapi

def quaternion_mul(quat_1, quat_2):
    """Multiply two quaternions.

    Args:
        quat_1 (isaacgym.gymapi.Quat): a quaternion of shape (4,).
        quat_2 (isaacgym.gymapi.Quat): a quaternion of shape (4,).

    Returns:
        isaacgym.gymapi.Quat: quat_1 * quat_2.
    """
    # Parse
    w1 = quat_1.w
    x1 = quat_1.x
    y1 = quat_1.y
    z1 = quat_1.z
    w2 = quat_2.w
    x2 = quat_2.x
    y2 = quat_2.y
    z2 = quat_2.z

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return gymapi.Quat(x, y, z, w)