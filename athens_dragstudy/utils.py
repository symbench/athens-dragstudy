import zipfile
import os
from athens_dragstudy import DATA_PATH, UAM_DATA, UAV_DATA, UAV_VEHICLES, UAM_VEHICLES, PRIMITIVE_DATA
from athens_dragstudy import CoffeeLessDragModel_old as cdmo
from athens_dragstudy import CoffeeLessDragModel as cdm
import json
import time
import random
import csv
import datetime
import matplotlib.pyplot as plt
import pandas as pd

uam_options = os.listdir(UAM_DATA)
uav_options = os.listdir(UAV_DATA)
prim_options = os.listdir(PRIMITIVE_DATA)

def plot_results():
    """ plot the desired results given a path to a result csv
        plot length param vs each of the 6 drag coefficients
    """
    filename = "/Users/michael/darpa/athens-dragstudy/athens_dragstudy/data/primitive/capsule/results/capsule_20221009-093959.csv"
    
    data = pd.read_csv(filename)

    print(data.columns)


    fig, axs = plt.subplots(3, 3)

    for d in data:
        print("(" in data[d])

    # min_tube_len = data['capsule_fuselage_TUBE_LENGTH'].min()
    # min_horz_diam = data['capsule_fuselage_HORZ_DIAMETER'].min()
    # min_vert_diam = data['capsule_fuselage_VERT_DIAMETER'].min()
    # min_dx = data['drag_x'].min()
    # min_dy = data['drag_y'].min()
    # min_dz = data['drag_z'].min()

    axs[0, 0].set_title("Tube length vs. drag x")
    axs[0, 0].scatter(data['capsule_fuselage_TUBE_LENGTH'], data['drag_x'], s=10)
    axs[0, 1].set_title("Tube length vs. drag y")
    axs[0, 1].scatter(data['capsule_fuselage_TUBE_LENGTH'], data['drag_y'], s=10)
    axs[0, 2].set_title("Tube length vs. drag z")
    axs[0, 2].scatter(data['capsule_fuselage_TUBE_LENGTH'], data['drag_z'], s=10)

    axs[1, 0].set_title("Horz Diameter vs. drag x")
    axs[1, 0].scatter(data['capsule_fuselage_HORZ_DIAMETER'], data['drag_x'], s=10, color='g')
    axs[1, 1].set_title("Horz Diameter vs. drag y")
    axs[1, 1].scatter(data['capsule_fuselage_HORZ_DIAMETER'], data['drag_y'], s=10, color='g')
    axs[1, 2].set_title("Horz Diameter vs. drag z")
    axs[1, 2].scatter(data['capsule_fuselage_HORZ_DIAMETER'], data['drag_z'], s=10, color='g')

    axs[2, 0].set_title("Vert Diameter vs. drag x")
    axs[2, 0].scatter(data['capsule_fuselage_VERT_DIAMETER'], data['drag_x'], s=10, color='m')
    axs[2, 1].set_title("Vert Diameter vs. drag y")
    axs[2, 1].scatter(data['capsule_fuselage_VERT_DIAMETER'], data['drag_y'], s=10, color='m')
    axs[2, 2].set_title("Vert Diameter vs. drag z")
    axs[2, 2].scatter(data['capsule_fuselage_VERT_DIAMETER'], data['drag_z'], s=10, color='m')


    plt.tight_layout()
    plt.show()




class DragRunner:
    def __init__(
        self,
        subject: str,
        num_runs: int,
        study_params: list,
        run_baseline: bool = False,
        from_zip: bool = False,
        primitive_struct: str = None,
        use_old_drag_model: bool = False
    ):

        assert subject in uav_options + uam_options + prim_options, "invalid drag subject"
        self.drag_subject = subject  # if primitive_struct is None else primitive_struct
        self.subject_type = None

        if self.drag_subject in uam_options:
            self.subject_type = "uam"
        elif self.drag_subject in uav_options:
            self.subject_type = "uav"
        else:
            self.subject_type = "primitive"
        
        #self.vehicle = self.drag_subject

        self.design_datafile = "designData.json"
        self.design_paramfile = "designParameters.json"

        # focus on length and wing parameters for now
        self.study_params = ["length"]  # if study_params is None else study_params

        self.num_runs = num_runs
        self.run_baseline = run_baseline
        #self.primitive_struct = primitive_struct


        # self.subject_type = None

        self.use_old_drag = use_old_drag_model


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

            self.subject_type = "uam"  # assume uam until proven otherwise
            for struct_name, prop_dict in self.paramdict.items():
                if "CADPART" in prop_dict and prop_dict["CADPART"] == "para_tube.prt":
                    self.subject_type = "uav"

            print(f"detected vehicle type: {self.subject_type}")
            if self.subject_type == "uam":
                self.BASE_PATH = os.path.join(UAM_DATA, self.vehicle)
            else:
                self.BASE_PATH = os.path.join(UAV_DATA, self.vehicle)

        # elif primitive_struct is not None:
        #     # get data from primitive folder and run drag
        #     print(f"primitive drag study of struct: {primitive_struct}")
        #     self.BASE_PATH = os.path.join(PRIMITIVE_DATA, primitive_struct)
        #     print(self.BASE_PATH)
        # elif self.vehicle:

        if self.subject_type == "uav":
            data = UAV_DATA
        elif self.subject_type == "uam":
            data = UAM_DATA
        else:
            data = PRIMITIVE_DATA

        self.BASE_PATH = os.path.join(data, self.drag_subject)
                
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

    def run_baseline_drag(self):

        if self.use_old_drag:
            result_id = str(hex(int(datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')))) + "_old"
        else:
            result_id = str(hex(int(datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')))) + "_new"
        
        if not os.path.exists(os.path.join(self.BASE_PATH, f"baseline_{result_id}")):
            os.makedirs(os.path.join(self.BASE_PATH, f"baseline_{result_id}"))
        else:
            if self.vehicle:
                print(
                    f"baseline results already exist for {self.vehicle} ({self.subject_type}) already exist, exiting"
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
        debug = True
        stl_output = True
        struct = True

        if self.use_old_drag:
            print("RUNNING OLD DRAG MODEL")
            drags, center, Con_Mesh, all_figs, spatial, parameter, prop = cdmo.run_full(
                self.datadict,   # datadict
                self.paramdict,  #
                include_wing,
                create_plot,
                debug,
                stl_output,
            )

            print("drags", drags)
            print("center", center)
            #print("type con mesh", type(Con_Mesh))
            #print("spatial", spatial)
            #print("parameter", parameter)
            #print("prop", prop)

            with open(os.path.join(self.BASE_PATH, f"baseline_{result_id}", "drag_center.txt"), "w") as f:
                f.write(str(drags) + "\n")
                f.write(str(center))
            print("wrote drag_center.txt")

            if stl_output:
                Con_Mesh.export(
                    os.path.join(self.BASE_PATH, f"baseline_{result_id}", "aircraft.stl")
                )
                print("exported aircraft.stl")

            if create_plot:
                for drxn, plot in list(zip(["x", "y", "z"], all_figs)):
                    plot.savefig(
                        os.path.join(
                            self.BASE_PATH, f"baseline_{result_id}", f"drag_plot_{drxn}.png"
                        )
                    )
                    print(f"wrote drag_plot_{drxn}.png")
            print("_____________OLD DRAG DONE_________________")
            return
        else:
            print("RUNNING NEW DRAG MODEL")
            drags, center, spatial, parameter, structure, prop, J_scale, T_scale = cdm.run_full(
                self.datafile,
                self.paramfile,
                include_wing,
                create_plot,
                debug,
                stl_output,
                struct
            )

            print("drags", drags)
            print("center", center)
            #print("spatial", spatial)
            #print("parameter", parameter)
            #print("structure", structure)
            #print("propo", prop)
            print("Jscale", J_scale)
            print("Tscale", T_scale)

            with open(os.path.join(self.BASE_PATH, f"baseline_{result_id}", "drag_center.txt"), "w") as f:
                f.write(str(drags) + "\n")
                f.write(str(center))
            print("wrote drag_center.txt")

            print("_________NEW DRAG DONE________")

    def run_dragmodel(self):
        """
        run the drag model after updating specified parameters
        """

        if self.run_baseline:  # run the analysis with the original parameters
            self.run_baseline_drag()
            self.use_old_drag = not self.use_old_drag
            self.run_baseline_drag()
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

            result_row = [self.drag_subject]
            new_params = self.update_drag_input_params()

            for structure in new_params:
                for param_name, param_val in new_params[structure].items():
                    if i == 0:  # only update header for first run
                        header_row.append(f"{structure}_{param_name}")
                    param_entries.append(param_val[1])  # the new value
            # print(f"running drag with updated params: {new_params}")

            struct = True  # obtain structural output
            drags, center, spatial, parameter ,structure, propo, J_scale, T_scale = cdm.run_full(
                self.datadict,  # datafile
                self.paramdict,  # paramfile
                include_wing,
                create_plot,
                debug,
                stl_output,
                struct
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
                str(self.drag_subject) + "_" + time.strftime("%Y%m%d-%H%M%S") + ".csv",
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

                if length_keys[self.drag_subject][0] in properties:
                    changed[structure] = {}

                    for length_key in length_keys[self.drag_subject]:
                        # if length_keys[self.drag_subject] in properties.keys():
                        #     length_key = length_keys[self.drag_subject]
                        #cad_part = properties["CADPART"]
                        orig_len = float(properties[length_key])
                        # print(f"modifying length of {length_key} from {orig_len}")
                        new_len = self.get_random_param_value(orig_len)

                        # print(f"orig_val is {orig_len} + {change} = {new_len}")
                        changed[structure][length_key] = (orig_len, new_len)
                        self.paramdict[structure][length_key] = new_len
            print(changed)
                        #break  # change only the first length, we assume the first length is the most important
       
        elif "wing" in self.study_params:

            wing_keys = {
                "wing": {

                }
            }
            # print("randomizing wing params")
            for structure, properties in self.paramdict.items():
                # print(f"structure is: {structure}")
                cadpart = (
                    "naca_sym_wing_taper.prt"
                    if self.subject_type == "uam"
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

if __name__ == "__main__":
    plot_results()