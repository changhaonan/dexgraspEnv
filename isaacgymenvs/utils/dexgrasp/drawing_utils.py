""" Provide drawing utilities for isaacgym. """
import math
from isaacgym import gymapi
from isaacgym import gymutil


def draw_6D_pose(gym, viewer, env, pos, rot):
    # Geometry
    axes_geom = gymutil.AxesGeometry(0.1)
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    sphere_geom = gymutil.WireframeSphereGeometry(
        0.03, 12, 12, sphere_pose, color=(1, 0, 0)
    )

    # Parse pose
    pose = gymapi.Transform()
    pose.p.x = pos[0]
    pose.p.y = pos[1]
    pose.p.z = pos[2]
    pose.r.x = rot[0]
    pose.r.y = rot[1]
    pose.r.z = rot[2]
    pose.r.w = rot[3]

    # Draw
    gymutil.draw_lines(axes_geom, gym, viewer, env, pose)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, pose)


def draw_bbox(gym, viewer, env, bbox):
    # Geometry
    bbox_geom = gymutil.WireframeBBoxGeometry(bbox, color=(1, 0, 0))

    # Draw
    pose = gymapi.Transform()
    gymutil.draw_lines(bbox_geom, gym, viewer, env, pose)
