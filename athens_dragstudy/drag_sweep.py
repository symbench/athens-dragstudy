import os
import time
import logging
import json
from pathlib import Path

from copy import deepcopy

import creopyson
from scipy.stats.qmc import LatinHypercube, scale
from creopyson import Client
from typing import Dict, Any, List
from pydantic import BaseModel, Field


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


class DesignInCreo:
    def __init__(self, config):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.creoson_client = Client(
            ip_adress=config["creoson_host"],
            port=config["creoson_port"]
        )
        self.design_parameters = config["parameter_map"]
        self.creoson_client.connect()
        self.assembly_name = self._open_assembly(config["assembly_path"])
        self.config = config

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

    def save_snapshot(self):
        pass

    def restart_creoson(self):
        pass

    def _get_component_path(self, all_paths, prt_name_):
        for details in all_paths:
            if details["file"] == prt_name_.upper():
                return details

    def to_design_data(self) -> List[ComponentData]:
        part_paths = self.creoson_client.bom_get_paths(paths=True, get_transforms=True)
        all_children_paths = part_paths['children']['children']
        design_data = []
        for component, component_param in self.design_parameters.items():
            self.logger.debug(f"Extracting transform matrices for {component}")
            try:
                massprops = self.creoson_client.file_massprops(file_=component + ".prt")
                component_transforms = self._get_component_path(all_children_paths, component + ".prt")
                mass = massprops["mass"]
                cg = [massprops["gravity_center"][axis] for axis in ["x", "y", "z"]]
                translation = [component_transforms["transform"]["origin"][axis] for axis in ["x", "y", "z"]]
                rot_x = [component_transforms["transform"][axis]["x"] for axis in ["x_axis", "y_axis", "z_axis"]]
                rot_y = [component_transforms["transform"][axis]["y"] for axis in ["x_axis", "y_axis", "z_axis"]]
                rot_z = [component_transforms["transform"][axis]["z"] for axis in ["x_axis", "y_axis", "z_axis"]]
                rot = [rot_x, rot_y, rot_z]
                component_data = ComponentData(
                    name=component,
                    cg=cg,
                    mass=mass,
                    translation=translation,
                    rotation=rot
                )
                design_data.append(component_data)
            except RuntimeError as e:
                self.logger.error(e)
                pass
        return design_data

    def propagate_parameters(self, changes, sample):
        assert len(changes) == len(sample)
        for all_component_prop_changes, value in zip(changes, sample):
            for comp_param_change in all_component_prop_changes:
                param = comp_param_change['param']
                component_name = comp_param_change['name']
                creo_prt_file = component_name + ".prt"

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

    def take_snapshot(self, save_dir):
        design_data_dict = self.design_data_dict()
        # parameters_dict = self.design_parameters_dict()
        with (save_dir / "DesignData.json").open("w") as json_file:
            json.dump(design_data_dict, json_file, indent=2)
            self.logger.debug(f"Saved design data in {save_dir / 'DesignData.json'}")

    def run(self):
        changes, lbounds, ubounds = self._get_parameters_to_change()
        sampler = LatinHypercube(d=len(ubounds), centered=True, seed=self.config["random_seed"])
        samples = sampler.random(self.config["num_samples"])
        samples = scale(sample=samples, l_bounds=lbounds, u_bounds=ubounds)
        ts = time.gmtime()
        save_dir = self.config["description"] + f"_{self.config['num_samples']}_on_" + time.strftime("%Y-%m-%d-%H-%M-%S", ts)
        self.logger.info(f"All output will be saved in {self.config['save_root']}/{save_dir}")
        save_dir = Path(self.config["save_root"]).resolve() / save_dir
        os.makedirs(save_dir, exist_ok=True)
        for j, sample in enumerate(samples):
            snapshot_dir = save_dir / f"{j + 1}"
            os.makedirs(snapshot_dir, exist_ok=True)
            try:
                self.propagate_parameters(changes, sample)
                self._regenerate_assembly()
                self.take_snapshot(snapshot_dir)
            except Exception as e:
                with (snapshot_dir / "err.txt").open('w') as err:
                    err.write(str(e))
                    self.logger.error(e)

    def design_data_dict(self) -> Dict[str, Any]:
        return {component_data.name: component_data.dict(by_alias=True) for component_data in self.to_design_data()}

    def _get_parameters_to_change(self):
        change_components = []
        lbounds = []
        ubounds = []
        for key, value in self.config["params"].items():
            assert ('min' in value and 'max' in value) or ('min' not in value or 'max' not in value)
            if 'min' in value and 'max' in value:
                multiple_changes = []
                for component_name in value["components"]:
                    multiple_changes.append({
                        "name": component_name,
                        "param": value["parameter"]
                    })
                lbounds.append(value["min"])
                ubounds.append(value["max"])
                change_components.append(multiple_changes)

        assert len(change_components) == len(lbounds) == len(ubounds)
        return change_components, lbounds, ubounds


if __name__ == "__main__":
    import yaml

    config_file = Path(__file__).resolve().parent.parent / "sweep_configs" / "fuseonly_config.yaml"
    with config_file.open("rb") as yaml_config:
        config_ = yaml.full_load(yaml_config.read().decode("utf-8"))
        with open(config_["parameter_map"], "rb") as param_map_json:
            config_["parameter_map"] = json.load(param_map_json)

        design = DesignInCreo(config_)
        design.run()
