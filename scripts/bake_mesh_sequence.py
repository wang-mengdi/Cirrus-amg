import os
import argparse
import numpy as np
import trimesh


def parse_slice(spec):
    parts = spec.split(":")
    parts = [int(p) if p != "" else None for p in parts]
    return slice(*parts)


def slice_to_frames(spec):
    s = parse_slice(spec)

    if s.stop is None:
        raise ValueError("Frame slice must specify stop value, e.g. 0:400")

    return list(range(s.stop))[s]


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


def bake_sequence(mesh_path, xform_dir, out_dir, frame_slice):

    mesh = load_mesh(mesh_path)

    frames = slice_to_frames(frame_slice)

    ext = os.path.splitext(mesh_path)[1].lower()

    if ext not in [".ply", ".obj"]:
        raise ValueError("Input mesh must be .ply or .obj")

    os.makedirs(out_dir, exist_ok=True)

    print("Input mesh:", mesh_path)
    print("Vertices:", len(mesh.vertices))
    print("Faces:", 0 if mesh.faces is None else len(mesh.faces))
    print("Output format:", ext)
    print("Total frames:", len(frames))

    for frame in frames:

        xform_file = os.path.join(xform_dir, f"transform_{frame:04d}.txt")

        if not os.path.exists(xform_file):
            raise FileNotFoundError(xform_file)

        M = read_transform_matrix(xform_file)

        baked = mesh.copy()
        baked.apply_transform(M)

        out_name = f"mesh_{frame:04d}{ext}"
        out_path = os.path.join(out_dir, out_name)

        baked.export(out_path)

        print(f"[{frame:04d}] -> {out_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mesh",
        help="input mesh (.ply or .obj)"
    )

    parser.add_argument(
        "--frames",
        required=True,
        help="frame slice, e.g. 0:401 or 0:401:2"
    )

    parser.add_argument(
        "--xform-dir",
        required=True,
        help="directory containing transform_XXXX.txt"
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="output directory"
    )

    args = parser.parse_args()

    bake_sequence(
        mesh_path=args.mesh,
        xform_dir=args.xform_dir,
        out_dir=args.out_dir,
        frame_slice=args.frames
    )


if __name__ == "__main__":
    main()
