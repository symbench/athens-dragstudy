import csv
from json import load
import os
import random
import sys

import creopyson.file
import creopyson.geometry
import creopyson.bom
import creopyson.parameter

from creopyson import Client
from numpy import isin
#from symbench_athens_client.models.uav_components import Propellers
# from symbench_athens_client.models.uam_components import Wings
# from symbench_athens_client.models.uam_components import Cylinders


from tqdm import tqdm
import time

prt_path = "C:\\JWork\\Agents\\uam-uav-models\\CAD"
uav_uam_models = os.listdir(prt_path)
uav_uam_model_lookup = {pf.split(".")[0] : os.path.join(prt_path, pf) for pf in uav_uam_models}
print(uav_uam_model_lookup)

capsule = uav_uam_model_lookup['fuse_capsule_new']
tube = uav_uam_model_lookup['para_tube']
cargo_case = uav_uam_model_lookup['para_cargo_case']

creo_client = Client(ip_adress="localhost", port=9056)
creo_client.connect()

class CreoManipulator:
    def __init__(self, component_repository="C:\\JWork\\Agents\\uam-uav-models\\CAD"):

        self.component_repository = component_repository
        self.component_models = os.listdir(self.component_repository)
        self.component_lookup = {x.split(".")[0] : os.path.join(self.component_lookup, x) for x in self.component_models}
        
        self.client = Client(ip_adress="localhost", port=9056)
        self.client.connect()



"""
creoson commands that are used in buildcad.py :

    get transforms: creopyson.file.get_transform(client, partpath)
    get paths of prts in an assembly: creopyson.bom.get_paths(client, asm_filename)


    creopyson.file.assemble(client, to_comp, parent_path, constraints)
    assemble the overall design by DFS of the graph (line 1300) buildtree fn
        ingredients: from_comp (prt), to_comp (prt), from_conn (str), to_conn (str)
            from_conn must be present on the current assembly, to_conn is a connector on the
            component to be added
        constraints refer to the a "type", an assembly ref (asmref) and a component ref (compref)

    set property of a component to a value: creopyson.parameter.set_(client, file_=part, name=prop_name, value=prop_value)
        used in buildcad functions setProperties and setParams

    get the mass properties of a component (includes volume, surface area, density, mass, cog, intertia,
    rotation matrix, rotation angles, radii of gyration wrt principal axes):
        creopyson.file.massprops(client, file_=partname (prt))

    get the bounding box of a component (x, y, z min max vals):
        creopyson.geometry.bound_box
    
    get the location of a component:
        creopyson.file.get_transform(client, part_path)
    
    regenerate a file:
        creopyson.file.regenerate(client)
    
    erase an assembly:
        creopyson.file.erase(client, erase_children=True)

    get file interferences:
        creopyson.file.interferences(client)


Goals here:

    - time each of these commands with some UAV components
    - try to connect a tube with a hub etc without using graphdb
    - obtain mass properties and transforms for components with varying params
"""

def get_transform(client, part_path):
    """ get the transforms for the passed part """
    s = time.time()
    result = creopyson.file.get_transform(client, part_path)
    e = time.time()
    print(f"get_transform time: {e-s}s returns type: {result}")
    return result

def set_property(client, part_path, prop_name, prop_value):
    """ set prop_name to prop_value for the part at part_path """
    s = time.time()
    result = creopyson.parameter.set_(client, file_=part_path, name=prop_name, value=prop_value)
    e = time.time()
    print(f"set_property time: {e-s}s returns type: {type(result), result}")
    return result

def get_bounding_box(part_path):
    """ get the x, y, z min and max for the bounding box of each component """
    s = time.time()
    result = creopyson.geometry.bound_box(part_path)
    e = time.time()
    print(f"get_bounding_box time: {e-s}s returns type: {type(result), result}")
    return result

def regenerate(client):
    """ update the current model e.g. after changing a parameter value """
    s = time.time()
    result = creopyson.file.regenerate(client)
    e = time.time()
    print(f"regenerate time: {e-s}s returns type: {type(result), result}")


def massprops(client, part_path):
    pass


def assemble(client, part_path):
    """ add a component to an existing model """
    s = time.time()
    result = creopyson.file.assemble(client, part_path)
    e = time.time()
    print(f"regenerate time: {e-s}s returns type: {type(result), result}")

def erase(client, erase_children=True):
    """ erase the current assembly """
    s = time.time()
    result = creopyson.file.erase(client, erase_childen=erase_children)
    e = time.time()
    print(f"regenerate time: {e-s}s returns type: {type(result), result}")


def get_parameter(base, interval_width, offset):
    """ """
    pass

def study_model(interval_width, include_transforms=True):
    """ """
    # get parameter value
    # update parameter value
    # regenerate
    # get massprops, transforms
    pass


#########


def massprops_to_csv_dict(massprops):
    """Return a csv style dict from creopyson mass-properties dict."""
    return {
        "surface_area": massprops["surface_area"],
        "density": massprops["density"],
        "mass": massprops["mass"],
        # Coordinate System
        "coordIxx": massprops["coord_sys_inertia_tensor"]["x_axis"]["x"],
        "coordIxy": massprops["coord_sys_inertia_tensor"]["x_axis"]["y"],
        "coordIxz": massprops["coord_sys_inertia_tensor"]["x_axis"]["z"],
        "coordIyx": massprops["coord_sys_inertia_tensor"]["y_axis"]["x"],
        "coordIyy": massprops["coord_sys_inertia_tensor"]["y_axis"]["y"],
        "coordIyz": massprops["coord_sys_inertia_tensor"]["y_axis"]["z"],
        "coordIzx": massprops["coord_sys_inertia_tensor"]["z_axis"]["x"],
        "coordIzy": massprops["coord_sys_inertia_tensor"]["z_axis"]["y"],
        "coordIzz": massprops["coord_sys_inertia_tensor"]["z_axis"]["z"],
        # Center of Gravity
        "cgIxx": massprops["ctr_grav_inertia_tensor"]["x_axis"]["x"],
        "cgIxy": massprops["ctr_grav_inertia_tensor"]["x_axis"]["y"],
        "cgIxz": massprops["ctr_grav_inertia_tensor"]["x_axis"]["z"],
        "cgIyx": massprops["ctr_grav_inertia_tensor"]["y_axis"]["x"],
        "cgIyy": massprops["ctr_grav_inertia_tensor"]["y_axis"]["y"],
        "cgIyz": massprops["ctr_grav_inertia_tensor"]["y_axis"]["z"],
        "cgIzx": massprops["ctr_grav_inertia_tensor"]["z_axis"]["x"],
        "cgIzy": massprops["ctr_grav_inertia_tensor"]["z_axis"]["y"],
        "cgIzz": massprops["ctr_grav_inertia_tensor"]["z_axis"]["z"],
    }


# get all wing .prt files in the corpus
def get_all_wings(client):
    return list(filter(lambda name: 'naca_sym_wing_taper.prt' in name, client.file_list()))


# read NACA profiles of all wings in the corpus 
def get_naca_profiles(wing_data_file):
    import pandas as pd
    dict_from_csv = pd.read_csv(wing_data_file).to_dict()
    profiles = [x.split(" ")[1] for x in dict_from_csv["Name"].values()]
    return profiles


def get_thickness_from_naca_profile(naca_profile, chord):
    #print(naca_profile)
    assert isinstance(naca_profile, str) and len(naca_profile) == 4
    #print(naca_profile, type(naca_profile))
    #print(chord, type(chord))
    return (int(naca_profile[2:]) / 100) * float(chord)  # thickness based on naca profile


#  given a tuple of wing properties from the pareto config, return the 
def get_wing_params_from_pareto_tuple(config):
    #  chord, span, load, profile
    return float(config[0]), float(config[1]), float(config[2]), config[3]


# input pareto_configs is a list of tuples of chord, span, load, profile
# return a generator of dictionaries of wing parameters
def get_configs_from_pareto_tuples(pareto_tuples):
    assert isinstance(pareto_tuples, list)
    assert isinstance(pareto_tuples[0], tuple)
    print(pareto_tuples)  # unpack to get chord, span, load 
    for ptuple in pareto_tuples:  # chord, span, load profile

        chord, span, load, profile = get_wing_params_from_pareto_tuple(ptuple)
        thickness = get_thickness_from_naca_profile(profile, chord)
        yield {'CHORD_1': chord,
                'CHORD_2': chord,
                'SPAN': span,
                'THICKNESS': thickness,
                'LOAD': load
            }


#  pareto_configs is a list of dictionaries with wing params set
def sample_wing_params(wing_data_file, samples=10, sample_around_pareto=False, pareto_tuples=None):
    #  user parameter ranges from swri sheet/guesses
    #TAPER_RANGE = [0, 4]  # unitless
    CHORD1_RANGE = [100, 2000]  # mm  assume chord1 = chord2 for now
    #CHORD2_RANGE = [100, 3000]  # mm
    SPAN_RANGE = [100, 6000]  # mm

    naca_profiles = get_naca_profiles(wing_data_file)
    taper_offset = 0  # dont' change for now

    if sample_around_pareto:
        assert isinstance(pareto_tuples, list) and isinstance(pareto_tuples[0], tuple)
        all_pconfigs = list(get_configs_from_pareto_tuples(pareto_tuples))

    # thickness is given by __XX digits of NACA profile as % of chord - symmetric or cambered
    
    count = 0
    while count < samples:

        if sample_around_pareto:
            pconfig_choice = random.choice(all_pconfigs)
            # only change chord and span; thickness is given by profile and load doesn't affect geometry
            prob = random.uniform(0, 1)
            if prob < 0.25: #  decrease chord, decrease span
                chord = pconfig_choice["CHORD_1"] - 100
                span = pconfig_choice["SPAN"] - 100
            elif prob > 0.25 and prob < 0.5: #  decrease chord, increase span
                chord = pconfig_choice["CHORD_1"] - 100
                span = pconfig_choice["SPAN"] + 100
            elif prob > 0.5 and prob < 0.75: #  increase chord, decrease span
                chord = pconfig_choice["CHORD_1"] + 100
                span = pconfig_choice["SPAN"] - 100
            else: #  prob >= 0.75, increase chord, increase span
                chord = pconfig_choice["CHORD_1"] + 100
                span = pconfig_choice["SPAN"] + 100
            thickness = pconfig_choice["THICKNESS"]  # stays the same 
            load = pconfig_choice["LOAD"] # stays the same
        else:
            profile = random.choice(naca_profiles)
            thickness = get_thickness_from_naca_profile(profile)
            chord = random.choice(list(range(CHORD1_RANGE[0], CHORD1_RANGE[1]+100, 100)))
            span = random.choice(list(range(SPAN_RANGE[0], SPAN_RANGE[1]+100, 100)))
            load = 8000.0 # default
        yield {'CHORD_1': chord,
                'CHORD_2': chord,
                'SPAN': span,
                'THICKNESS': thickness,
                'LOAD': load
               }
        count +=1
        print(f"Progress: {count}/{samples}", end="\r")



def assign_wing_params(client, wings, model_fname, config):
    """
    config is a list of dictionaries:

        [{CHORD_1: x, CHORD_2: y, SPAN: z, THICKNESS: w, TAPER_OFFSET: v}, ...]        
    """
    #  config = [{'CHORD_1': 100.0, 'CHORD_2': 100.0, 'SPAN': 500.0, 'THICKNESS': 10.0, 'TAPER_OFFSET': 0.0}]
    
    assert isinstance(config, dict)
    wing_file = wings[0]  #  hack to get prt file since these is only one wing
    USER_PARAMS = ["CHORD_1", "CHORD_2", "SPAN", "THICKNESS", "LOAD"]  #, "TAPER_OFFSET"]
    params = client.parameter_list(name="*", file_= wing_file)
    #print(f"wing params: {params}")
    for param in params:
        if param["name"] in USER_PARAMS:
            param_val = config[param["name"]]
            #print(f"setting {param['name']} to {param_val}")
            client.parameter_set(
                name=param["name"],
                value=param_val,
                type_=param["type"],
                file_=param["owner_name"]
            )

    client.file_regenerate(file_=wing_file)
    params = client.parameter_list(name="*", file_=wing_file)
    creopyson.file.regenerate(client=client, file_=model_fname)
    massprops = client.file_massprops(file_=model_fname)
    component = {}
    random_wing = random.choice(Wings)  #  not actually random
    wing_dict = random_wing.dict(by_alias=True, exclude_none=True)
    component["Component"] = random_wing.name
    for param in params:
        if param["name"] in wing_dict:
            component[f'Component_param_{param["name"]}'] = param["value"]


    csv_massprops = massprops_to_csv_dict(massprops=massprops)
    csv_massprops.update(component)
    return csv_massprops


# main entry point
# configs is either a list of dictionaries of the form [{CHORD_1: x, CHORD_2: y, SPAN: z, THICKNESS: w, TAPER_OFFSET: v}, ...]
# OR a list of tuples of the form [(chord1, chord2, span, profile), ...] for thickness to be derived
def write_data(creo_client, write_fname, model_fname, configs=None):
    wing_part = get_all_wings(client=creo_client)
    with open(write_fname + ".csv", "w") as file:
        row_dicts = []
        for config in tqdm(configs):
            #print(f"setting {config}")
            row_dicts.append(assign_wing_params(creo_client, wing_part, model_fname, config))
        print(row_dicts)
        writer = csv.DictWriter(file, fieldnames=row_dicts[0].keys())
        writer.writeheader()
        writer.writerows(row_dicts)
    print(f"wrote {write_fname}")



def run(dname, fname, configs, write_fname):
    print("in run")
    creo_client = Client(
        ip_adress="localhost",
        port=9056
    )

    creo_client.connect()
    print("connected")

    #print(f"configs type is {type(configs)}")

    creo_client.file_open(
       fname,
       dirname=dname
    #os.getcwd() + "\\" + "TestBench_CADTB_V1"
    )

    print(f"opened file {fname}")
    
    write_data(creo_client, write_fname, fname, configs)


#  parse the pareto wing data and return a list of tuples containing (chord, span, load, profile)
def read_wing_pareto_data(fname):
    with open(fname, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield((row['chord'], row['span'], row['flight_load'], row['profile'].split(" ")[1]))


def do_wing_study(dname):
    
    fname = dname + "\\" + "naca_sym_wing_taper.prt"
    write_fname = "C:" + "\\" + "wing_massprops_neg_drag_pos_lift_max_weight_30" #"wing_massprops_neg_drag_pos_vol_sample_100_1"
    wing_data_file = "C:" + "\\" + "Wing.csv"

    wing_pareto_file = "C:" + "\\" + "uav-analysis" + "\\" + "uav_analysis" + "\\" + "data_hackathon2_uam" + "\\" +  'wing_analysis_neg_drag_pos_lift_max_weight_30.csv' # "wing_analysis_neg_drag_pos_available_volume.csv" # "wing_analysis_rear_pareto.csv"

    print("reading wing params from file")
    ptuples = list(read_wing_pareto_data(wing_pareto_file))
    
    #  reading from file
    print("makeing configs")
    configs = list(get_configs_from_pareto_tuples(ptuples))
    print(f"total configs: {len(configs)}")

    #  sample around pareto
    #configs = list(sample_wing_params(wing_data_file, samples=100, sample_around_pareto=True, pareto_tuples=ptuples))

    print("running")
    run(dname, fname, configs, write_fname)




def write_data_cylinder(creo_client, write_fname, model_fname, configs=None):
    with open(write_fname + ".csv", "w") as file:
        row_dicts = []
        for config in tqdm(configs):
            print(f"setting {config}")
            row_dicts.append(assign_cylinder_params(creo_client, model_fname, config))
        print(row_dicts)
        writer = csv.DictWriter(file, fieldnames=row_dicts[0].keys())
        writer.writeheader()
        writer.writerows(row_dicts)
    print(f"wrote {write_fname}")

def run_cylinder(dname, fname, configs, write_fname):
    print("in run")
    creo_client = Client(
        ip_adress="localhost",
        port=9056
    )

    creo_client.connect()
    print("connected")

    #print(f"configs type is {type(configs)}")

    creo_client.file_open(
       fname,
       dirname=dname
    #os.getcwd() + "\\" + "TestBench_CADTB_V1"
    )

    print(f"opened file {fname}")
    
    write_data_cylinder(creo_client, write_fname, fname, configs)


def make_cylinder_configs():

    # pthick <= diameter - 1 
    is_valid = lambda x: x[0] <= x[1] - 1

    for lngth in range(1000, 1500, 20): #range(100, 1000, 20):
        for diam in range(200, 400, 20): #range(40, 200, 20):
            for pthick in range(100, 200, 20): #range(10, 100, 20):
                if is_valid((pthick, diam)):
                    yield {'DIAMETER': diam, 'PORT_THICKNESS': pthick, 'LENGTH': lngth}
    # Diameter 40, 200
    # port thickness 5, 200
    # length 0 to 1000
    # yield dictionary of configs changing diameter 6 to 200, port thickness 0 to 200

def do_cylinder_study(dname, idx=0, samples=50):
    fname = dname + "\\" + "fuse_cyl_ported.prt"
    write_loc = "C:" + "\\" + "uav-analysis" + "\\" + "uav_analysis" + "\\" + "data_hackathon2_uam" + "\\"
    write_fname = write_loc + f"cylinder_mass_props_xl_{idx+1}_diff_test_with-patch"
    
    configs = list(make_cylinder_configs())
    print(f"number of all configs: {len(configs)}, current idx: {idx}")

    print(f"running idx {idx} - increment each time")
    run_cylinder(dname, fname, configs[idx * samples: (idx * samples) + samples], write_fname)




def assign_cylinder_params(client, model_fname, config):
    """
    config is a list of dictionaries:

        [{CHORD_1: x, CHORD_2: y, SPAN: z, THICKNESS: w, TAPER_OFFSET: v}, ...]        
    """
    #  config = [{'CHORD_1': 100.0, 'CHORD_2': 100.0, 'SPAN': 500.0, 'THICKNESS': 10.0, 'TAPER_OFFSET': 0.0}]
    
    assert isinstance(config, dict)
    #print(f"current config {config}")
    USER_PARAMS = ["LENGTH", "DIAMETER", "PORT_THICKNESS"]  #, "TAPER_OFFSET"]
    params = client.parameter_list(name="*", file_= model_fname)
    #print(f"wing params: {params}")
    for param in params:
        if param["name"] in USER_PARAMS:
            param_val = config[param["name"]]
            #print(f"setting {param['name']} to {param_val}")
            client.parameter_set(
                name=param["name"],
                value=param_val,
                type_=param["type"],
                file_=param["owner_name"]
            )
    try:
        client.file_regenerate(file_=model_fname)
        params = client.parameter_list(name="*", file_=model_fname)
        creopyson.file.regenerate(client=client, file_=model_fname)
        massprops = client.file_massprops(file_=model_fname)
        component = {}
        random_cylinder = random.choice(Cylinders)  #  not actually random
        wing_dict = random_cylinder.dict(by_alias=True, exclude_none=True)
        component["Component"] = random_cylinder.name
        for param in params:
            if param["name"] in wing_dict:
                component[f'Component_param_{param["name"]}'] = param["value"]
        
        csv_massprops = massprops_to_csv_dict(massprops=massprops)
        csv_massprops.update(component)
    except ValueError as ve:
        print(f"regeneration error for {config}, skipping...")
    if csv_massprops:
        return csv_massprops
    else:
        return {"REGEN ERROR": None for _ in range(21)}

   


if __name__ == "__main__":
    print(uav_uam_model_lookup)
    #dname = "C:" + "\\" + "UAM_CAD"
    
    #idx = 2  # up to 8 for normal sizes, up to 7 for xl
    # 
    # if sys.argv[1]:
    #     idx = int(sys.argv[1])
    #     do_cylinder_study(dname, idx=idx)