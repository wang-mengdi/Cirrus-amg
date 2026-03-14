import os
import re
import argparse
import numpy as np
import trimesh


BOX_MIN = np.array([0.0, 0.0, 0.0])
BOX_MAX = np.array([1.0, 1.0, 2.0])


def read_transform_matrix(path):
    M = np.loadtxt(path)
    if M.shape != (4, 4):
        raise ValueError(f"{path} is not a 4x4 matrix")
    return M


def load_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Failed to load mesh")

    return mesh


def find_frames(xform_dir):
    pattern = re.compile(r"transform_(\d+)\.txt")
    frames = []

    for f in os.listdir(xform_dir):
        m = pattern.match(f)
        if m:
            frames.append(int(m.group(1)))

    if not frames:
        raise RuntimeError("No transform_XXXX.txt files found")

    frames.sort()
    return frames


def point_box_signed_margin(points):
    points = np.asarray(points)

    d = np.column_stack([
        points[:, 0] - BOX_MIN[0],
        BOX_MAX[0] - points[:, 0],
        points[:, 1] - BOX_MIN[1],
        BOX_MAX[1] - points[:, 1],
        points[:, 2] - BOX_MIN[2],
        BOX_MAX[2] - points[:, 2],
    ])

    return np.min(d, axis=1)


def apply_transform(points, M):
    ones = np.ones((points.shape[0], 1))
    homo = np.concatenate([points, ones], axis=1)
    out = homo @ M.T
    return out[:, :3]


def mesh_max_radius(mesh):
    v = np.asarray(mesh.vertices)
    return np.linalg.norm(v, axis=1).max()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("mesh")
    parser.add_argument("--xform-dir", required=True)

    args = parser.parse_args()

    mesh = load_mesh(args.mesh)
    verts = np.asarray(mesh.vertices)

    frames = find_frames(args.xform_dir)

    max_radius = mesh_max_radius(mesh)

    mesh_min_margin = np.inf
    traj_min_margin = np.inf

    for frame in frames:

        path = os.path.join(args.xform_dir, f"transform_{frame:04d}.txt")

        M = read_transform_matrix(path)

        # mesh transform
        verts_world = apply_transform(verts, M)
        margins = point_box_signed_margin(verts_world)
        mesh_min_margin = min(mesh_min_margin, margins.min())

        # trajectory
        p = apply_transform(np.array([[0.0, 0.0, 0.0]]), M)
        traj_margin = point_box_signed_margin(p)[0]
        traj_min_margin = min(traj_min_margin, traj_margin)

    print("mesh_max_radius =", max_radius)
    print("mesh_min_box_margin =", mesh_min_margin)
    print("trajectory_min_box_margin =", traj_min_margin)


if __name__ == "__main__":
    main()