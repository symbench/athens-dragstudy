import json
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

from athens_dragstudy.CoffeeLessDragModel import ellipticaldrag

vp = 1  # 5  # 30 m/s
ap = 1  # 4  # +0 deg
mp = 100  # 50#scale of arrows? 50 for UAM, 100 for UAV works well

vel = np.arange(15, 31, 15)  # np.arange(5, 51, 5)  # Range of velocties for table
mu = 18 * 10**-6  # air viscosity
rho = 1.225  # air density
ang = np.arange(-20, 1, 20)  # np.arange(-20, 21, 50)

Vel = np.transpose(np.tile(vel, [len(ang), 1]))


def transformation_matrix(design_data, direction="x"):
    assert direction in {"x", "y", "z"}
    Trans = np.array(design_data["Translation"])
    CG = np.array(design_data["CG"])
    Trans = Trans.astype(float)
    CG = CG.astype(float)
    CG = CG + Trans
    Trans = np.vstack(Trans)
    Rot = np.array(design_data["Rotation"])
    Rot = Rot.astype(float)
    # Rot=Rot.transpose()

    Tform = np.hstack((Rot, Trans))

    Tform = np.vstack((Tform, (0, 0, 0, 1)))
    if direction == "x":
        rotr = trimesh.transformations.rotation_matrix(
            np.deg2rad(0), [1, 0, 0]
        )  # should be -90
        Tform = trimesh.transformations.concatenate_matrices(rotr, Tform)

    elif direction == "y":
        rotr = trimesh.transformations.rotation_matrix(
            np.deg2rad(-90), [0, 0, 1]
        )  # should be -90
        Tform = trimesh.transformations.concatenate_matrices(rotr, Tform)
    elif direction == "z":
        rotr = trimesh.transformations.rotation_matrix(
            np.deg2rad(-90), [0, 1, 0]
        )  # should be -90
        Tform = trimesh.transformations.concatenate_matrices(rotr, Tform)

    return Tform


from scipy.spatial.transform import Rotation as R


def _calculate_drag_params(
    horizontal_diameter,
    vertical_diameter,
    bottom_connector_rotation,
    cyl_length,
    direction="x",
):
    diafuse = (horizontal_diameter + vertical_diameter) / 2

    if direction == "x":
        axis_of_symmetry = [1, 0, 0]
    elif direction == "y":
        axis_of_symmetry = [0, 1, 0]
    elif direction == "z":
        axis_of_symmetry = [0, 0, 1]

    rot_angle = 90 - bottom_connector_rotation
    mat = R.from_euler("z", rot_angle, degrees=True)
    cap_norm = mat.apply(np.array(axis_of_symmetry))

    drag_params = {
        "ell_chord": (float(cyl_length) + diafuse) / 1000,
        "ell_len": (float(cyl_length) + diafuse) / 1000,
        "ell_dia": diafuse / 1000,
        "ell_n": np.abs(cap_norm),
        "ang": ang,
        "vel": vel,
        "mu": mu,
        "rho": rho,
    }

    return drag_params


def ellipticaldrag_params_without_creo(fuses_params, direction="x"):

    assert direction in {"x", "y", "z"}
    drag_params = {}
    for name, fuse_params in fuses_params.items():
        horizontal_diameter = fuse_params["HORZ_DIAMETER"]
        vertical_diameter = fuse_params["VERT_DIAMETER"]
        cyl_length = fuse_params.get("FUSE_CYL_LENGTH", fuses_params.get("TUBE_LENGTH"))
        bottom_connector_rotation = fuse_params["BOTTOM_CONNECTOR_ROTATION"]
        drag_params[name] = _calculate_drag_params(
            horizontal_diameter,
            vertical_diameter,
            bottom_connector_rotation,
            cyl_length,
            direction,
        )

    return drag_params


def ellipticaldrag_params(fuses_params, fuses_data, direction="x"):
    drag_params = {}
    for (fuse_name, fuse_params), (_, fuse_data) in zip(
        fuses_params.items(), fuses_data.items()
    ):
        Tform = transformation_matrix(fuse_data, direction=direction)
        diafuse = 0.5 * (
            float(fuse_params["HORZ_DIAMETER"]) + float(fuse_params["VERT_DIAMETER"])
        )
        mesh2cad = [
            [np.cos(3 * np.pi / 2), 0, np.sin(3 * np.pi / 2), 0],
            [0, 1, 0, 0],
            [-np.sin(3 * np.pi / 2), 0, np.cos(3 * np.pi / 2), 0],
            [0, 0, 0, 1],
        ]

        cyl_length = fuse_params.get("FUSE_CYL_LENGTH", fuses_params.get("TUBE_LENGTH"))

        mesh = trimesh.creation.cylinder(
            diafuse / 2,
            height=float(cyl_length),
            transform=mesh2cad,
            center_mass=fuse_data["CG"],
            sections=10,
        )
        mesh.apply_transform(Tform)
        cap_norm = abs(mesh.symmetry_axis)
        cap_cg = mesh.center_mass

        mesh = trimesh.creation.capsule(
            height=float(cyl_length), radius=diafuse / 2, count=[10, 10]
        )
        mesh.apply_transform(mesh2cad)
        mesh.apply_transform(Tform)

        mesh.center_mass = cap_cg

        drag_params[fuse_name] = {
            "ell_chord": (float(cyl_length) + diafuse) / 1000,
            "ell_len": (float(cyl_length) + diafuse) / 1000,
            "ell_dia": diafuse / 1000,
            "ell_n": cap_norm,
            "ang": ang,
            "vel": vel,
            "mu": mu,
            "rho": rho,
        }
    return drag_params


def run(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        "DragExploration", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "drag-func", choices=["elliptical-drag"], help="Which drag function to run"
    )
    parser.add_argument("--exp-dir", help="The experiment directory to run it from")
    parser.add_argument("--direction", help="The drag direction", type=str, default="x")
    args = parser.parse_args(args)

    exp_dir = Path(args.exp_dir).resolve()
    params = pd.read_csv(exp_dir / "params.csv")
    if getattr(args, "drag-func") == "elliptical-drag":
        for index, row in params.iterrows():
            parent = Path(row["files_location"]).resolve()
            design_data = parent / "designData.json"
            design_params = parent / "designParameters.json"

            assert design_data.exists() and design_params.exists()

            with design_data.open("r") as json_file:
                design_data = json.load(json_file)

            with design_params.open("r") as json_file:
                design_params = json.load(json_file)

            fuselages = list(
                filter(
                    lambda p: "fuse" in design_params[p].get("CADPART", ""),
                    design_params,
                )
            )
            all_fuselages_params = {fuse: design_params[fuse] for fuse in fuselages}
            all_fuselages_data = {fuse: design_data[fuse] for fuse in fuselages}

            elliptical_drag_params = ellipticaldrag_params(
                all_fuselages_params, all_fuselages_data, args.direction
            )
            elliptical_drag_params_copy = ellipticaldrag_params_without_creo(
                all_fuselages_params, args.direction
            )
            for key, value in elliptical_drag_params.items():
                print(value, elliptical_drag_params_copy[key])
                assert np.allclose(
                    value["ell_n"], elliptical_drag_params_copy[key]["ell_n"]
                )
                cd, cl, cf, warea = ellipticaldrag(**value)
                rarea = warea * np.ones([1, np.size(ang)])
                mfun = np.ones([100, Vel.shape[1]])
                modder = np.min(mfun, axis=0)
                drag = (
                    0.5
                    * rho
                    * Vel**2
                    * cd
                    * np.tile(rarea, [np.size(Vel, axis=0), 1])
                    * np.tile(modder, [len(vel), 1])
                )
                print(f"{parent.name}, Drag_{args.direction} => {drag[1, 1]}")
