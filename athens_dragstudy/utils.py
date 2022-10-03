import zipfile
import os
from athens_dragstudy import DATA_PATH, UAM_DATA, UAV_DATA, UAV_VEHICLES, UAM_VEHICLES, PRIMITIVE_DATA
from athens_dragstudy import CoffeeLessDragModel as cdm
import json
import time
import random
import csv


def plot_results():
    """ plot the desired results given a path to a result csv
        plot length param vs each of the 6 drag coefficients
    """ 
    pass

class DragRunner:
    def __init__(
        self,
        vehicle: str,
        num_runs: int,
        study_params: list,
        run_baseline: bool = False,
        from_zip: bool = False,
        primitive_struct: str = None
    ):

        self.drag_subject = vehicle if primitive_struct is None else primitive_struct
        self.vehicle = self.drag_subject

        self.design_datafile = "designData.json"
        self.design_paramfile = "designParameters.json"

        # focus on length and wing parameters for now
        self.study_params = ["length"] if study_params is None else study_params

        self.num_runs = num_runs
        self.run_baseline = run_baseline
        self.primitive_struct = primitive_struct
        self.vehicle_type = None


        if from_zip:  # read json data directly from jenkins build artifact
            # get the param and json from the zip file

            # when we are able to run UAVs through the pipeline, update this code

            z = zipfile.ZipFile(
                os.path.join(DATA_PATH, "design_zips", self.vehicle + ".zip")
            )
            with z.open("archive/result_1/designData.json") as f:
                self.datadict = json.loads(f.read().decode("utf-8"))
            # print(type(self.datadict))
            with z.open("archive/result_1/designParameters.json") as f:
                self.paramdict = json.loads(f.read().decode("utf-8"))
            # print(type(self.paramdict))

            self.vehicle_type = "uam"  # assume uam until proven otherwise
            for struct_name, prop_dict in self.paramdict.items():
                if "CADPART" in prop_dict and prop_dict["CADPART"] == "para_tube.prt":
                    self.vehicle_type = "uav"

            print(f"detected vehicle type: {self.vehicle_type}")
            if self.vehicle_type == "uam":
                self.BASE_PATH = os.path.join(UAM_DATA, self.vehicle)
            else:
                self.BASE_PATH = os.path.join(UAV_DATA, self.vehicle)

        elif primitive_struct is not None:
            # get data from primitive folder and run drag
            print(f"primitive drag study of struct: {primitive_struct}")
            self.BASE_PATH = os.path.join(PRIMITIVE_DATA, primitive_struct)
            print(self.BASE_PATH)
        elif self.vehicle:

            if self.vehicle in UAV_VEHICLES:
                self.vehicle_type = "uav"
                data = UAV_DATA
            elif self.vehicle in UAM_VEHICLES:
                self.vehicle_type = "uam"
                data = UAM_DATA
            self.BASE_PATH = os.path.join(data, self.vehicle)
                
        self.datafile = os.path.join(self.BASE_PATH, self.design_datafile)
        self.paramfile = os.path.join(self.BASE_PATH, self.design_paramfile)
        self.datadict = json.loads(open(self.datafile).read())
        self.paramdict = json.loads(open(self.paramfile).read())
        assert self.datadict
        assert self.paramdict

    def set_params_and_run_drag(self):
        """Run the drag model with a specific set of parameters that come from
        the parameter drag dataset object"""

        run_params = {"leg": 95, "arm": 220}

        for name in self.paramdict.keys():
            item = name.split("_")[0]
            if item in ["arm", "leg"]:
                # print(
                #     f"changing {name} from {self.paramdict[name]['LENGTH']} to {float(run_params[item])}"
                # )
                self.paramdict[name]["LENGTH"] = float(run_params[item])
                print(name, self.paramdict[name]["LENGTH"])

        include_wing = False
        create_plot = False
        debug = False
        stl_output = False
        drags, center, stl_mesh, plots = cdm.run_full(
            self.datadict, self.paramdict, include_wing, create_plot, debug, stl_output
        )

        if not os.path.exists(os.path.join(self.BASE_PATH, "tmp")):
            os.makedirs(os.path.join(self.BASE_PATH, "tmp"))

        if stl_output:
            stl_mesh.export(os.path.join(self.BASE_PATH, "tmp", "aircraft.stl"))

        print(f"drags: {drags}")
        print(f"centers: {center}")

    def run_dragmodel(self):
        """
        run the drag model after updating specified parameters
        """

        if self.run_baseline:  # run the analysis with the original parameters
            if not os.path.exists(os.path.join(self.BASE_PATH, "baseline")):
                os.makedirs(os.path.join(self.BASE_PATH, "baseline"))
            else:
                if self.vehicle:
                    print(
                        f"baseline results already exist for {self.vehicle} ({self.vehicle_type}) already exist, exiting"
                    )
                    return
                elif self.primitive_struct:
                    print(
                        f"baseline results already exist for {self.primitive_struct} already exist, exiting"
                    )
                    return

            print("running baseline drag model")
            include_wing = True
            create_plot = True
            debug = False
            stl_output = True

            drags, center, stl_mesh, plots = cdm.run_full(
                self.datadict,
                self.paramdict,
                include_wing,
                create_plot,
                debug,
                stl_output,
            )

            print(drags)
            print(center)

            with open(
                os.path.join(self.BASE_PATH, "baseline", "drag_center.txt"), "w"
            ) as f:
                f.write(str(drags) + "\n")
                f.write(str(center))
            print("wrote drag_center.txt")

            if stl_output:
                stl_mesh.export(
                    os.path.join(self.BASE_PATH, "baseline", "aircraft.stl")
                )
                print("exported aircraft.stl")

            if create_plot:
                for drxn, plot in list(zip(["x", "y", "z"], plots)):
                    plot.savefig(
                        os.path.join(
                            self.BASE_PATH, "baseline", f"drag_plot_{drxn}.png"
                        )
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
                for param_name, param_val in new_params[structure].items():
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
        print(f"{self.num_runs} done in {total_time}")

        if not os.path.exists(os.path.join(self.BASE_PATH, "results")):
            os.makedirs(os.path.join(self.BASE_PATH, "results"))
        with open(
            os.path.join(
                self.BASE_PATH,
                "results",
                str(self.vehicle) + "_" + time.strftime("%Y%m%d-%H%M%S") + ".csv",
            ),
            "w",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(header_row)
            writer.writerows(all_results)

    def write_json(self):
        with open(self.paramfile, "w") as f:
            json.dump(self.paramdict, f, indent=3)

    def get_random_param_value(self, orig_val, scale=20):
        direction = -1 if bool(round(random.random())) else 1
        change = direction * random.random() * scale
        return orig_val + change


    def update_drag_input_params(self, scale=20):

        changed = {}  # { structure: (old_param, new_param) }

        if "length" in self.study_params:
            # print("randomizing length params")

            length_keys = {"capsule": ("TUBE_LENGTH", "HORZ_DIAMETER", "VERT_DIAMETER"),
                           "tube": ("LENGTH")}
            for structure, properties in self.paramdict.items():

                for length_key in length_keys[self.drag_subject]:
                    # if length_keys[self.drag_subject] in properties.keys():
                    #     length_key = length_keys[self.drag_subject]
                        #cad_part = properties["CADPART"]
                        orig_len = float(properties[length_key])
                        # print(f"modifying length of {length_key} from {orig_len}")
                        new_len = self.get_random_param_value(orig_len)

                        # print(f"orig_val is {orig_len} + {change} = {new_len}")
                        changed[structure] = {length_key: (orig_len, new_len)}
                        self.paramdict[structure][length_key] = new_len
                        #break  # change only the first length, we assume the first length is the most important
       
        elif "wing" in self.study_params:
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
                    # print(f"modifying wing span and chord of {structure}")
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