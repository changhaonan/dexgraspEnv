""" Provide drawing utilities for isaacgym. """
import math
from isaacgym import gymapi
from isaacgym import gymutil


def draw_6D_pose(gym, viewer, env, pos, rot, sphere_radius=0.03, axis_length=0.1):
    # Geometry
    axes_geom = gymutil.AxesGeometry(axis_length)
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    sphere_geom = gymutil.WireframeSphereGeometry(
        sphere_radius, 12, 12, sphere_pose, color=(1, 0, 0)
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


def draw_3D_pose(gym, viewer, env, pos, sphere_radius=0.03, color=(1, 0, 0)):
    # Geometry
    sphere_rot = gymapi.Quat.from_euler_zyx(0, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    sphere_geom = gymutil.WireframeSphereGeometry(
        sphere_radius, 12, 12, sphere_pose, color
    )

    # Parse pose
    pose = gymapi.Transform()
    pose.p.x = pos[0]
    pose.p.y = pos[1]
    pose.p.z = pos[2]

    # Draw
    gymutil.draw_lines(sphere_geom, gym, viewer, env, pose)


def draw_bbox(gym, viewer, env, bbox):
    # Geometry
    bbox_geom = gymutil.WireframeBBoxGeometry(bbox, color=(1, 0, 0))

    # Draw
    pose = gymapi.Transform()
    gymutil.draw_lines(bbox_geom, gym, viewer, env, pose)
