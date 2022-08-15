# Copyright (C) 2022, Michael Sandborn
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import argparse
import sys

import matplotlib.pyplot as plt
from typing import Optional
from athens_dragstudy import CoffeeLessDragModel as cdm
from athens_dragstudy import fit
import time
import csv
import os
import json
import random
import time

UAM_DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.dirname(__file__), "data/uam")
)
UAV_DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.dirname(__file__), "data/uav")
)


# TODO: if you want a new vehicle to study it must go here
UAM_VEHICLES = ["spade", "tbar_full", "tbar_double", "tbar_single", "tbar_minime", "tbar_steerwing"]
UAV_VEHICLES = ["axe", "quad"]
VALID_VEHICLES = UAM_VEHICLES + UAV_VEHICLES

print(UAV_DATA, UAM_DATA)


class VehicleMeshBuilder:
    """
    get the drag model code that creates a vehicle mesh from the data and design parameters
    the meshes alone might be interesting later
    """

    def __init__(self, vehicle):
        pass
        # assert vehicle in VALID_VEHICLES, f"vehicle {vehicle} is invalid! valid vehicles are: {VALID_VEHICLES}"

    def build(self):
        pass


class DragRunner:
    def __init__(
        self,
        vehicle: str,
        num_runs: int,
        study_params: list,
        run_baseline: bool = False,
    ):

        assert (
            vehicle in VALID_VEHICLES
        ), f"vehicle {vehicle} is invalid! valid vehicles are: {VALID_VEHICLES}"
        self.vehicle = vehicle

        self.study_params = ["length"] if study_params is None else study_params
        # assert self.study_params == ["length"] or \
        #        self.study_params == ["wing"] or \
        #        self.study_params == ["prop"] or \
        #        self.study_params == ["all"], f"invalid entry for study_params! Must be [length | wing | prop | all]"

        self.num_runs = num_runs

        if self.vehicle == "axe" or self.vehicle == "quad":
            self.vehicle_type = "uav"
            self.BASE_PATH = os.path.join(UAV_DATA, self.vehicle)
            self.datafile = os.path.join(self.BASE_PATH, "designData.json")
            self.paramfile = os.path.join(self.BASE_PATH, "designParameters.json")
        else:
            self.vehicle_type = "uam"
            self.BASE_PATH = os.path.join(UAM_DATA, self.vehicle)
            self.datafile = os.path.join(self.BASE_PATH, "designData.json")
            self.paramfile = os.path.join(self.BASE_PATH, "designParameters.json")

        self.datadict = json.loads(open(self.datafile).read())
        assert self.datadict
        self.paramdict = json.loads(open(self.paramfile).read())
        assert self.paramdict
        self.run_baseline = run_baseline

    def run_dragmodel(self):
        """
        run the drag model after updating specified parameters
        """

        if self.run_baseline:  # run the analysis with the original parameters
            if not os.path.exists(os.path.join(self.BASE_PATH, "baseline")):
                os.mkdir(os.path.join(self.BASE_PATH, "baseline"))
            else:
                print(
                    f"baseline results for {self.vehicle} ({self.vehicle_type}) already exist, exiting"
                )
                return

            print(f"running baseline drag model")
            drags, center, stl_mesh, plots = cdm.run_full(
                self.datafile,
                self.paramfile,
                True,  # include wing
                True,  # create plot
                False,  # debug
                True,
            )  # stl output

            with open(
                os.path.join(self.BASE_PATH, "baseline", "drag_center.txt"), "w"
            ) as f:
                f.write(str(drags) + "\n")
                f.write(str(center))
            print("wrote drag_center.txt")
            stl_mesh.export(os.path.join(self.BASE_PATH, "baseline", "aircraft.stl"))
            print("exported aircraft.stl")

            for drxn, plot in list(zip(["x", "y", "z"], plots)):
                plot.savefig(
                    os.path.join(self.BASE_PATH, "baseline", f"drag_plot_{drxn}.png")
                )
                print(f"wrote drag_plot_{drxn}.png")
            return

        all_results = []

        header_row = [
            "vehicle",
            "drag_x",
            "drag_y",
            "drag_z",
            "drag_cx",
            "drag_cy",
            "drag_cz",
        ]

        total_time = 0
        for i in range(self.num_runs):
            if (i + 1) % 100 == 0 and i != 0:
                print(
                    f"avg runtime after {i+1} iters: {round(total_time / (i+1), 3)}s/it"
                )

            s = time.time()

            include_wing = True
            create_plot = False
            debug = True
            stl_output = False

            param_entries = []

            result_row = [self.vehicle]
            new_params = self.update_drag_input_params()

            for structure in new_params:
                # print(new_params[structure])
                for param_name, param_val in new_params[structure].items():
                    # print(param_name, param_val)
                    if i == 0:  # only update header for first run
                        header_row.append(f"{structure}_{param_name}")
                    param_entries.append(param_val[1])  # the new value
            # print(f"running drag with updated params: {new_params}")

            drags, center, spatial, parameter, prop = cdm.run_full(
                self.datadict,
                self.paramdict,
                include_wing,
                create_plot,
                debug,
                stl_output,
            )
            result_row.extend(drags)
            result_row.extend(center)
            result_row.extend(param_entries)

            e = time.time()
            total_time += e - s

            # print(f"[TIME TAKEN] {e-s}s for iteration {i+1}/{self.num_runs}")
            # print(f"run {i+1}/{self.num_runs}")
            # print("drags")
            # print(drags)
            # print("center")
            # print(center)

            # print("result row")
            # print(result_row)

            all_results.append(result_row)

        if not os.path.exists(os.path.join(self.BASE_PATH, "results")):
            os.mkdir(os.path.join(self.BASE_PATH, "results"))
        with open(
            os.path.join(
                self.BASE_PATH,
                "results",
                self.vehicle + "_" + time.strftime("%Y%m%d-%H%M%S") + ".csv",
            ),
            "w",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(header_row)
            writer.writerows(all_results)

    def write_json(self):
        with open(self.paramfile, "w") as f:
            json.dump(self.paramdict, f, indent=3)

    def update_drag_input_params(self, scale=20):

        changed = {}  # { structure: (old_param, new_param) }

        if "length" in self.study_params:
            # print("randomizing length params")
            for structure, properties in self.paramdict.items():
                # print(structure)
                cadpart = (
                    "fuse_cyl_ported.prt"
                    if self.vehicle_type == "uam"
                    else "para_tube.prt"
                )
                if (
                    "LENGTH" in properties.keys() and properties["CADPART"] == cadpart
                ):  # only change length for now
                    orig_len = properties["LENGTH"]
                    # print(f"modifying length of {structure} from {orig_len}")
                    direction = -1 if bool(round(random.random())) else 1
                    change = direction * random.random() * scale
                    new_len = orig_len + change
                    # print(f"orig_val is {orig_len} + {change} = {new_len}")
                    changed[structure] = {"LENGTH": (orig_len, new_len)}
                    self.paramdict[structure]["LENGTH"] += change
        if "prop" in self.study_params:
            # print("randomizing prop params")
            for structure, properties in self.paramdict.items():
                if "PROP_TYPE" in properties.keys():  # only change diameter for now
                    # print(f"modifying diameter of {structure} from {properties['DIAMETER']}")
                    orig_diam = float(properties["DIAMETER"])
                    direction = -1 if bool(round(random.random())) else 1
                    change = direction * random.random() * scale
                    new_diam = orig_diam + change
                    # print(f"orig_val is {orig_diam} + {change} = {new_diam}")
                    changed[structure] = {"DIAMETER": (orig_diam, new_diam)}
                    self.paramdict[structure]["DIAMETER"] = str(new_diam)
        if "wing" in self.study_params:
            # print("randomizing wing params")
            for structure, properties in self.paramdict.items():
                # print(f"structure is: {structure}")
                cadpart = (
                    "naca_sym_wing_taper.prt"
                    if self.vehicle_type == "uam"
                    else "uav_wing.prt"
                )
                if (
                    "NACA_PROFILE" in properties.keys()
                    and properties["CADPART"] == cadpart
                ):
                    # only change chords and span of symmetric wings for now, later thickness and taper
                    #print(f"modifying wing span and chord of {structure}")
                    orig_c1 = properties["CHORD_1"]
                    orig_c2 = properties["CHORD_2"]
                    orig_span = properties["SPAN"]

                    names = ["CHORD_1", "CHORD_2", "SPAN"]
                    for idx, orig_val in enumerate([orig_c1, orig_c2, orig_span]):
                        direction = -1 if bool(round(random.random())) else 1
                        change = direction * random.random() * scale
                        new_val = orig_val + change
                        # print(f"wing param is {names[idx]}")
                        # print(f"orig_val is {orig_val} + {change} = {new_val}")
                        if structure in changed:
                            # print(
                            #    f"updating {structure} params from {orig_val} to {new_val}"
                            # )
                            changed[structure][names[idx]] = (orig_val, new_val)
                        else:
                            # print(
                            #    f"init {structure} params with {orig_val} and {new_val}"
                            # )
                            changed[structure] = {names[idx]: (orig_val, new_val)}
                        self.paramdict[structure][names[idx]] = new_val
        # self.write_json()
        return changed


def run():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument('--corpus', type=str, metavar='corpus type',
    #                     choices=['uav', 'uam'])
    parser.add_argument(
        "--vehicle",
        type=str,
        metavar="design name",
        help="the name of the vehicle of interest",
    )
    parser.add_argument(
        "--fit", action="store_true", help="fit the specified vehicle to"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="run the drag model for the original vehicle parameters",
    )
    parser.add_argument(
        "--rand-length",
        action="store_true",
        help="randomly change connector lengths in the design by a small amount",
    )
    parser.add_argument(
        "--rand-wing",
        action="store_true",
        help="randomly change connector lengths in the design by a small amount",
    )
    parser.add_argument(
        "--rand-prop",
        action="store_true",
        help="randomly change connector lengths in the design by a small amount",
    )
    parser.add_argument("--runs", type=int, help="the number of drag runs to complete")

    args = parser.parse_args()
    print(args)

    if args.fit and args.vehicle:
        fit.run(args.vehicle)
        sys.exit(0)
    elif args.fit and not args.vehicle:
        print("need to specify a vehicle to fit its data")
        sys.exit(0)

    study_params = []
    if args.rand_length:
        study_params.append("length")
    if args.rand_prop:
        study_params.append("prop")
    if args.rand_wing:
        study_params.append("wing")
    print(f"study params are: {study_params}")
    dr = DragRunner(args.vehicle, args.runs, study_params, args.baseline)
    dr.run_dragmodel()


if __name__ == "__main__":
    run()
