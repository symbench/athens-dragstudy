import json
import logging
import os
import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from creopyson import Client
from pydantic import BaseModel, Field
from scipy.stats.qmc import LatinHypercube, scale

from athens_dragstudy.CoffeeLessDragModel import run_full


def get_logger(name, level=logging.DEBUG):
    """Get a logger instance."""
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(ch)

    return logger


class ComponentData(BaseModel):
    name: str = Field(..., alias="Name")
    cg: List[float] = Field(..., alias="CG")
    mass: float = Field(..., alias="MASS")
    translation: List[float] = Field(..., alias="Translation")
    rotation: List[List[float]] = Field(..., alias="Rotation")

    class Config:
        allow_population_by_field_name = True


class DesignSweep:
    """A design sweep using CREO.

    Provided a configuration for sweep, this class runs LHC sampling
    in the provided parameter ranges and runs a sweep of the parameters
    for that design with CREO in loop to produce necessary JSON files to
    run the drag study. This class is specific to UAV corpus and relies on
    data coming out of `direct2cad` workflow from SWRI.

    parameters:
    ----------
    config: dict, required=True
        The configuration for this sweep.
    """

    def __init__(self, config):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.setLevel(config.get("loglevel", logging.DEBUG))
        self.creoson_client = Client(
            ip_adress=config["creoson_host"], port=config["creoson_port"]
        )
        self.design_parameters = self._load_design_params(config["parameter_map"])
        self.creoson_client.connect()
        self.assembly_name = self._open_assembly(config["assembly_path"])
        self.config = config

    @staticmethod
    def _load_design_params(design_params_loc):
        with open(design_params_loc, "rb") as param_map_json:
            return json.load(param_map_json)

    def _open_assembly(self, assembly_path):
        """Given an assembly path, open it in CREO"""
        file_path = Path(assembly_path).resolve()
        assembly_dir = str(file_path.parent)
        assembly_name = file_path.name
        self.creoson_client.file_open(
            file_=assembly_name,
            dirname=assembly_dir,
        )
        self.logger.info(f"Opened design {assembly_name} from {assembly_dir} in CREO")
        return assembly_name

    def restart_creoson(self):
        """Restart creoson server in case of failures"""
        pass

    def _get_component_path(self, all_paths, prt_name):
        """Get the component patch by its part name"""
        for details in all_paths:
            if details["file"] == prt_name.upper():
                return details

    def to_design_data(self) -> List[ComponentData]:
        """Returns list with necessary rotation/translation/CG terms for every component in the design."""
        part_paths = self.creoson_client.bom_get_paths(paths=True, get_transforms=True)
        all_children_paths = part_paths["children"]["children"]
        design_data = []
        for component, component_param in self.design_parameters.items():
            self.logger.debug(f"Extracting transform matrices for {component}")
            try:
                massprops = self.creoson_client.file_massprops(file_=component + ".prt")
                component_transforms = self._get_component_path(
                    all_children_paths, component + ".prt"
                )
                mass = massprops["mass"]
                cg = [massprops["gravity_center"][axis] for axis in ["x", "y", "z"]]
                translation = [
                    component_transforms["transform"]["origin"][axis]
                    for axis in ["x", "y", "z"]
                ]
                rot_x = [
                    component_transforms["transform"][axis]["x"]
                    for axis in ["x_axis", "y_axis", "z_axis"]
                ]
                rot_y = [
                    component_transforms["transform"][axis]["y"]
                    for axis in ["x_axis", "y_axis", "z_axis"]
                ]
                rot_z = [
                    component_transforms["transform"][axis]["z"]
                    for axis in ["x_axis", "y_axis", "z_axis"]
                ]
                rot = [rot_x, rot_y, rot_z]
                component_data = ComponentData(
                    name=component,
                    cg=cg,
                    mass=mass,
                    translation=translation,
                    rotation=rot,
                )
                design_data.append(component_data)
            except RuntimeError as e:
                self.logger.error(e)
                pass
        return design_data

    def propagate_parameters(self, changes, sample):
        """Propagate parameters for every change from the sample"""
        assert len(changes) == len(sample)
        for all_component_prop_changes, value in zip(changes, sample):
            for comp_param_change in all_component_prop_changes:
                param = comp_param_change["param"]
                component_name = comp_param_change["name"]
                creo_prt_file = component_name + ".prt"
                if creo_prt_file == "BatteryController.prt":  # Quirk with the design
                    continue
                creo_param = self.creoson_client.parameter_list(
                    name=param, file_=creo_prt_file
                ).pop()
                self.creoson_client.parameter_set(
                    name=creo_param["name"],
                    value=value,
                    file_=creo_prt_file,
                    type_=creo_param["type"],
                    no_create=True,
                )
                self.logger.info(
                    f"Set {param} for {component_name} "
                    f"in CREO ( {creo_prt_file}'s {param} = {value} )"
                )

    def _regenerate_assembly(self):
        """Regenerate the currently loaded assembly in CREO."""
        self.creoson_client.file_regenerate(file_=self.assembly_name)

    def set_original_params(self):
        """Reset the design to its original form in CREO"""
        for comp_name, params in self.design_parameters.items():
            prt_file = f"{comp_name}.prt"
            cad_params = self.creoson_client.parameter_list(file_=prt_file)
            for cad_param in cad_params:
                if (name := cad_param["name"]) in params:
                    self.creoson_client.parameter_set(
                        name=name, value=params[name], file_=prt_file
                    )
                    self.logger.debug(
                        f"Set parameter {name} for {comp_name} to {params[name]}"
                    )
        self._regenerate_assembly()
        self.logger.info(
            "Successfully regenerated assembly after setting original design parameters"
        )

    def run_drag_model(self, save_dir):
        design_data_dict = self.design_data_dict()
        design_params_dict = self.design_parameters_dict()

        with (save_dir / "DesignData.json").open("w") as json_file:
            json.dump(design_data_dict, json_file, indent=2)
            self.logger.debug(f"Saved design data in {save_dir / 'DesignData.json'}")

        with (save_dir / "designParameters.json").open("w") as json_file:
            json.dump(design_params_dict, json_file, indent=2)
            self.logger.debug(
                f"Saved design data in {save_dir / 'designParameters.json'}"
            )

        drags, center, spatial, parameter, structure, prop, J_scale, T_scale = run_full(
            DataName=design_data_dict,
            ParaName=design_params_dict,
            include_wing=True,
            create_plot=True,
            debug=True,
            stl_output=True,
            struct=True,
            save_dir=save_dir,
        )

        return drags, center

    def sweep(self):
        """Run the sweep with LHC sampling of the parameter ranges"""
        changes, lbounds, ubounds = self._sweep_info()
        sampler = LatinHypercube(
            d=len(ubounds), centered=True, seed=self.config["random_seed"]
        )
        samples = sampler.random(self.config["num_samples"])
        samples = scale(sample=samples, l_bounds=lbounds, u_bounds=ubounds)
        ts = time.localtime()
        save_dir = (
            self.config["description"]
            + f"_{self.config['num_samples']}_on_"
            + time.strftime("%Y-%m-%d-%H-%M-%S", ts)
        )
        self.logger.info(
            f"All output will be saved in {self.config['save_root']}/{save_dir}"
        )
        save_dir = Path(self.config["save_root"]).resolve() / save_dir
        os.makedirs(save_dir, exist_ok=True)
        with (config_path := save_dir / "config.yaml").open("w") as yaml_file:
            yaml.dump(self.config, yaml_file)
            self.logger.info(f"Config saved in {config_path}")
        param_list = []
        for change_dicts in changes:
            for change_dict in change_dicts:
                param_list.append(change_dict["name"] + "_" + change_dict["param"])
        params_df = pd.DataFrame(
            columns=param_list
            + ["x_fuse", "y_fuse", "z_fuse", "X_fuseuu", "Y_fusevv", "Z_fuseww", "files_location"]
        )
        for j, sample in enumerate(samples):
            snapshot_dir = save_dir / f"{j + 1}"
            os.makedirs(snapshot_dir, exist_ok=True)
            try:
                self.propagate_parameters(changes, sample)
                self._regenerate_assembly()
                drags, centers = self.run_drag_model(snapshot_dir)
                params_df.loc[len(params_df)] = list(sample) + [
                    *centers,
                    *drags,
                    snapshot_dir.resolve(),
                ]
            except Exception as e:
                with (snapshot_dir / "err.txt").open("w") as err:
                    err.write(str(e))
                    self.logger.error(e)
            if j % 50 == 0:
                params_df.to_csv(save_dir / "params.csv")

        params_df.to_csv(save_dir / "params.csv")
        self.set_original_params()

    def design_data_dict(self) -> Dict[str, Any]:
        """Returns dict with necessary rotation/translation/CG terms for every component in the design."""
        return {
            component_data.name: component_data.dict(by_alias=True)
            for component_data in self.to_design_data()
        }

    def design_parameters_dict(self):
        """Return a dictionary of design parameters for the design currently active in CREO."""
        params_copy = deepcopy(self.design_parameters)
        for name, params in params_copy.items():
            creo_params = self.creoson_client.parameter_list(file_=name + ".prt")
            for creo_param in creo_params:
                if creo_param["name"] in params:
                    params[creo_param["name"]] = creo_param["value"]

        return params_copy

    def _sweep_info(self):
        """Get information on components/parameters to change for the sweep using current config."""
        change_components = []
        lbounds = []
        ubounds = []
        for key, value in self.config["params"].items():
            assert ("min" in value and "max" in value) or (
                "min" not in value or "max" not in value
            )
            if "min" in value and "max" in value:
                target_components = []
                for component_name in value["components"]:
                    target_components.append(
                        {"name": component_name, "param": value["parameter"]}
                    )
                lbounds.append(value["min"])
                ubounds.append(value["max"])
                change_components.append(target_components)

        assert len(change_components) == len(lbounds) == len(ubounds)
        return change_components, lbounds, ubounds


def run(args):
    parser = ArgumentParser(
        description="Sweep a design in CREO to generate JSON files required and run the drag model"
    )
    parser.add_argument("--config-file", help="The sweep configuration file", type=str)
    parser.add_argument("--verbose", help="The verbosity of logs", action="store_true")

    args = parser.parse_args(args)
    config_file = Path(args.config_file)
    with config_file.open("rb") as yaml_config:
        config_ = yaml.full_load(yaml_config.read().decode("utf-8"))
        if args.verbose:
            config_["loglevel"] = "DEBUG"
        design = DesignSweep(config_)
        design.sweep()


if __name__ == "__main__":
    import sys

    run(sys.argv[1:])
