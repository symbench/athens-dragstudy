import copy
import json
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.stats.qmc import LatinHypercube, scale

from athens_dragstudy.CoffeeLessDragModel import ellipticaldrag
from athens_dragstudy.utils import list_float

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


def _calculate_drag_params(
    horizontal_diameter,
    vertical_diameter,
    bottom_connector_rotation,
    cyl_length,
    direction="x",
):
    diafuse = (horizontal_diameter + vertical_diameter) / 2
    rot_angle = bottom_connector_rotation
    mat = R.from_euler("z", rot_angle, degrees=True)
    Tform = np.hstack((mat.as_matrix(), np.vstack([0, 0, 0])))

    Tform = np.vstack((Tform, (0, 0, 0, 1)))

    # if direction == "x":
    #     axis_of_symmetry = [1, 0, 0]
    # elif direction == "y":
    #     axis_of_symmetry = [0, 1, 0]
    # elif direction == "z":
    #     axis_of_symmetry = [0, 0, 1]

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

    mesh2cad = [
        [np.cos(3 * np.pi / 2), 0, np.sin(3 * np.pi / 2), 0],
        [0, 1, 0, 0],
        [-np.sin(3 * np.pi / 2), 0, np.cos(3 * np.pi / 2), 0],
        [0, 0, 0, 1],
    ]

    mesh = trimesh.creation.cylinder(
        diafuse / 2,
        height=float(cyl_length),
        transform=mesh2cad,
        center_mass=[0, 0, 0],
        sections=10,
    )
    mesh.apply_transform(Tform)
    cap_norm = abs(mesh.symmetry_axis)
    cap_cg = mesh.center_mass
    drag_params = {
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


def ellipticaldrag_params_without_creo(fuses_params: Dict, direction="x") -> Dict:

    assert direction in {"x", "y", "z"}
    drag_params = {}
    for name, fuse_params in fuses_params.items():
        horizontal_diameter = fuse_params["HORZ_DIAMETER"]
        vertical_diameter = fuse_params["VERT_DIAMETER"]
        cyl_length = fuse_params.get("FUSE_CYL_LENGTH", fuse_params.get("TUBE_LENGTH"))
        bottom_connector_rotation = fuse_params["BOTTOM_CONNECTOR_ROTATION"]
        drag_params[name] = _calculate_drag_params(
            horizontal_diameter,
            vertical_diameter,
            bottom_connector_rotation,
            cyl_length,
            direction,
        )

    return drag_params


def ellipticaldrag_params(fuses_params: Dict, fuses_data: Dict, direction="x") -> Dict:
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

        cyl_length = fuse_params.get("FUSE_CYL_LENGTH", fuse_params.get("TUBE_LENGTH"))

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


def force_at_reference_velocity(params: Dict) -> np.ndarray:
    cd, cl, cf, warea = ellipticaldrag(**params)
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
    return drag


def verify_fuselage_drag(params: pd.DataFrame, direction="x") -> None:
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
            all_fuselages_params, all_fuselages_data, direction
        )
        elliptical_drag_params_copy = ellipticaldrag_params_without_creo(
            all_fuselages_params, direction
        )
        for key, value in elliptical_drag_params.items():
            assert np.allclose(
                value["ell_n"], elliptical_drag_params_copy[key]["ell_n"]
            )
            drag = force_at_reference_velocity(value)
            print(f"{parent}, Drag_{direction} at 30 m/sec => {drag[1, 1]} N")


def sample_fuselage_drag(save_dir, samples=100, seed=42):
    save_dir = Path(save_dir).resolve()
    if not save_dir.exists():
        os.makedirs(save_dir)

    params = {
        "BOTTOM_CONNECTOR_ROTATION": (0, 360),
        "VERT_DIAMETER": (70, 170),
        "HORZ_DIAMETER": (140, 240),
        "FUSE_CYL_LENGTH": (100, 300),
    }
    lbounds = []
    ubounds = []
    for key, value in params.items():
        lbounds.append(value[0])
        ubounds.append(value[1])

    sampler = LatinHypercube(d=len(ubounds), centered=True, seed=seed)
    samples = sampler.random(samples)
    samples = scale(sample=samples, l_bounds=lbounds, u_bounds=ubounds)
    drags = []
    for sample in samples:
        sample_dict = {key: sample[i] for i, key in enumerate(params)}
        print(f"Calculating fuselage drag for \n {json.dumps(sample_dict, indent=2)}")
        for d in ["x", "y", "z"]:
            drag_params = ellipticaldrag_params_without_creo(
                {"fuselage": sample_dict}, direction=d
            )
            df = force_at_reference_velocity(drag_params["fuselage"])
            sample_dict[f"drag_{d}"] = df[1, 1]
        drags.append(sample_dict)
    drags_df = pd.DataFrame.from_records(drags)
    drags_df.to_csv(save_dir / "sample-elliptical-drag.csv", index=False)


def fuselage_drag_single_run(params, direction="all"):
    forces = copy.deepcopy(params)
    if direction == "all":
        directions = list("xyz")
    else:
        directions = list(direction)
    for d in directions:
        drag_params = ellipticaldrag_params_without_creo(
            {"fuselage": params}, direction=d
        )
        f = force_at_reference_velocity(drag_params["fuselage"])
        forces[f"drag_force_{d}"] = f[1, 1]
    return forces


def run(args=None):

    parser = ArgumentParser(
        "DragExploration", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=[
            "fuselage-drag-from-experiment",
            "sample-fuselage-drag",
            "fuselage-drag-single-run",
            "sweep-all-rotations",
        ],
        help="Which drag function to explore run",
    )
    parser.add_argument(
        "--exp-dir",
        help="The experiment directory/save directory",
        required="fuselage-drag-from-experiment" in sys.argv
        or "sweep-all-rotations" in sys.argv,
    )
    parser.add_argument("--drag-direction", help="The drag direction", type=str)
    parser.add_argument(
        "--fuselage-parameters",
        type=list_float,
        required="fuselage-drag-single-run" in sys.argv,
        help="parameters as comma separated list VERT_DIAMETER,HORZ_DIAMETER,CYL_LENGTH,BOTTOM_CONNECTOR_ROTATION",
        metavar="VERT_DIAMETER,HORZ_DIAMETER,CYL_LENGTH,BOTTOM_CONNECTOR_ROTATION",
    )
    parser.add_argument(
        "--save-dir",
        help="where to save the output files",
        required="sample-fuselage-drag" in sys.argv
        or "sweep-all-rotations" in sys.argv,
    )
    parser.add_argument(
        "--sample-size",
        "-n",
        help="Number of samples to generate",
        type=int,
        required="sample-fuselage-drag" in sys.argv,
    )
    parser.add_argument("--random-seed", type=int, help="The random seed", default=42)
    args = parser.parse_args(args)

    if getattr(args, "command") == "fuselage-drag-from-experiment":
        exp_dir = Path(args.exp_dir).resolve()
        params = pd.read_csv(exp_dir / "params.csv")
        verify_fuselage_drag(params, args.drag_direction)

    elif getattr(args, "command") == "sample-fuselage-drag":
        sample_fuselage_drag(args.save_dir, args.sample_size, args.random_seed)

    elif getattr(args, "command") == "fuselage-drag-single-run":
        params = args.fuselage_parameters
        params = {
            "VERT_DIAMETER": params[0],
            "HORZ_DIAMETER": params[1],
            "FUSE_CYL_LENGTH": params[2],
            "BOTTOM_CONNECTOR_ROTATION": params[3],
        }
        forces = fuselage_drag_single_run(params, args.drag_direction or "all")
        print(f"Drag at 30m/sec in direction = {json.dumps(forces, indent=2)}")

    elif getattr(args, "command") == "sweep-all-rotations":
        parent = Path(args.exp_dir).resolve()
        root_params = parent / "0" / "designParameters.json"
        with root_params.open("r") as params_file:
            params = json.load(params_file)

            for component, component_params in params.items():
                if "fuse_capsule_new" in component_params.get("CADPART", ""):
                    fuselage_params = {
                        "VERT_DIAMETER": component_params["VERT_DIAMETER"],
                        "HORZ_DIAMETER": component_params["HORZ_DIAMETER"],
                        "FUSE_CYL_LENGTH": component_params["FUSE_CYL_LENGTH"],
                        "BOTTOM_CONNECTOR_ROTATION": component_params[
                            "BOTTOM_CONNECTOR_ROTATION"
                        ],
                    }
                    drags = []
                    for j in range(0, 360):
                        fuselage_params["BOTTOM_CONNECTOR_ROTATION"] = float(j)
                        drags.append(fuselage_drag_single_run(fuselage_params, "all"))

                if len(drags) == 0:
                    raise ValueError("Capsule fuselage not found")
                drags_df = pd.DataFrame.from_records(drags)
                drags_df.to_csv(
                    Path(args.save_dir) / "all-rotation-sweeps.csv", index=False
                )


if __name__ == "__main__":
    run(sys.argv)
