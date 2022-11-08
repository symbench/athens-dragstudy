import csv
from http import client
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

from scipy.stats.qmc import LatinHypercube, scale

from tqdm import tqdm
import time

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
    result = creopyson.file.get_transform(client, path=part_path)
    e = time.time()
    print(f"get_transform time: {e-s}s returns type: {result}")
    return result

def set_property(client, part_path, prop_name, prop_value):
    """ set prop_name to prop_value for the part at part_path """
    s = time.time()
    result = creopyson.parameter.set_(client, file_=part_path, name=prop_name, value=prop_value)
    e = time.time()
    #print(f"set_property time: {e-s}s returns type: {type(result), result}")
    return result

def get_bounding_box(client, part_path):
    """ get the x, y, z min and max for the bounding box of each component """
    s = time.time()
    active = creopyson.file.get_active(client)
    print(active)
    result = creopyson.geometry.bound_box(client)
    e = time.time()
    print(f"get_bounding_box time: {e-s}s returns type: {type(result), result}")
    return result

def regenerate(client):
    """ update the current model e.g. after changing a parameter value """
    s = time.time()
    result = creopyson.file.regenerate(client)
    e = time.time()
    print(f"regenerate time: {e-s}s returns type: {type(result), result}")


def get_massprops(client, part_path):
    s = time.time()
    result = creopyson.file.massprops(client, file_=part_path)
    e = time.time()
    print(f"massprops: {e-s}s returns type: {type(result), result}")
    return result

def select_part(part_dict):
    """ return a random part from a list of parts """
    part = random.choice(list(part_dict.keys()))
    return part_dict[part]

def select_connection(part_type):
    """ given a part type, select a connection on the part to connect """
    pass

def get_paths(client, tforms=True, paths=True):
    s = time.time()
    result = creopyson.bom.get_paths(client, paths=paths, get_transforms=tforms)
    e = time.time()
    print(f"regenerate time: {e-s}s returns type: {type(result), result}")
    return result

def assemble(client, part_path, constraints=None, path=None):
    """ add a component to an existing model """
    s = time.time()
    result = creopyson.file.assemble(client, part_path, constraints=constraints, path=path)
    e = time.time()
    print(f"regenerate time: {e-s}s returns type: {type(result), result}")

def erase(client, erase_children=True):
    """ erase the current assembly """
    s = time.time()
    result = creopyson.file.erase(client, erase_childen=erase_children)
    e = time.time()
    print(f"regenerate time: {e-s}s returns type: {type(result), result}")


def collect_massprops(subject, num_samples=3000, with_tforms=False):
    assert subject in ["capsule", "uavwing"], f"invalid massprops subject {subject}"  # new components to study must go here

    header = ["subject", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "surface_area", "cg_x", "cg_y",
              "cg_z", "density", "mass", "volume", "cg_inert_tensor_x", "cg_inert_tensor_y", "cg_inert_tensor_z"]

    print(f"sweeping massprops for {subject}")

    if with_tforms:
        fname = f"./{subject}_bbox_mprops_params.csv"
    else:
        fname = f"./{subject}_bbox_mprops_params_floor_ht.csv"

    if subject == "uavwing":

        header.extend(["CHORD_1", "CHORD_2", "SPAN"])

        with open(fname, "a") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            # for hd in range(200, 1000, 50):  # HORZ_DIAMETER
            #     for vd in range(100, 1000, 50): # VERT_DIAMETER
            #         for tl in range(100, 1000, 50):  # TUBE_LENGTH
                        # set_property(creo_client, capsule, "HORZ_DIAMETER", hd)
                        # set_property(creo_client, capsule, "VERT_DIAMETER", vd)
                        # set_property(creo_client, capsule, "TUBE_LENGTH", tl)
            for cd in range(500, 820, 20):  # CHORD
                for sp in range(200, 1220, 20): # SPAN
                        set_property(creo_client, uavwing, "CHORD_1", cd)
                        set_property(creo_client, uavwing, "CHORD_2", cd)
                        set_property(creo_client, uavwing, "SPAN", sp)
                        regenerate(creo_client)
                        if with_tforms:
                            tforms = get_transform(creo_client, uavwing)
                            print(tforms)
                        bbox = get_bounding_box(creo_client, uavwing)
                        mprops = get_massprops(creo_client, uavwing)
                        bbox.update(mprops)
                        
                        if with_tforms:
                            bbox.update(tforms)

                        #print(bbox)
                        res = {
                            'subject': 'uavwing',
                            'xmin': bbox['xmin'],
                            'xmax': bbox['xmax'],
                            'ymin': bbox['ymin'],
                            'ymax': bbox['ymax'],
                            'zmin': bbox['zmin'],
                            'zmax': bbox['zmax'],
                            'surface_area': mprops['surface_area'],
                            'cg_x': mprops['gravity_center']['x'],
                            'cg_y': mprops['gravity_center']['y'],
                            'cg_z': mprops['gravity_center']['z'],
                            'density': mprops['density'],
                            'mass': mprops['mass'],
                            'volume': mprops['volume'],
                            'cg_inert_tensor_x': mprops['ctr_grav_inertia_tensor']['x_axis']['x'],
                            'cg_inert_tensor_y': mprops['ctr_grav_inertia_tensor']['y_axis']['y'],
                            'cg_inert_tensor_z': mprops['ctr_grav_inertia_tensor']['z_axis']['z'],
                            'CHORD_1': cd,
                            'CHORD_2': cd,
                            'SPAN': sp
                        }

                        print(f"CHORD_1: {cd}, CHORD_2: {cd}, SPAN: {sp}")

                        writer.writerow(res)
    elif subject == "capsule":

        header.extend(["HORZ_DIAMETER", "VERT_DIAMETER", "FUSE_CYL_LENGTH", "FLOOR_HEIGHT"])

        with open(fname, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            lbounds = [60, 60, 70, 3]
            ubounds = [700, 700, 1000, 30]
            sampler = LatinHypercube(d=len(ubounds), centered=True, seed=42)
            samples = sampler.random(num_samples)
            samples = scale(samples, l_bounds=lbounds, u_bounds=ubounds) 

            for sample in samples:
                hd, vd, tl, fh = sample
                set_property(creo_client, capsule, "HORZ_DIAMETER", hd)
                set_property(creo_client, capsule, "VERT_DIAMETER", vd)
                set_property(creo_client, capsule, "FUSE_CYL_LENGTH", tl)
                set_property(creo_client, capsule, "FLOOR_HEIGHT", fh)
                try:
                    regenerate(creo_client)
                except RuntimeError as e:
                    print(f"Error. Msg <=> {e}, sample: [hd, vd, tl, fh] = {sample}")
                    continue
                            
                bbox = get_bounding_box(creo_client, capsule)
                mprops = get_massprops(creo_client, capsule)

                bbox.update(mprops)

                res = {
                    'subject': subject,
                    'xmin': bbox['xmin'],
                    'xmax': bbox['xmax'],
                    'ymin': bbox['ymin'],
                    'ymax': bbox['ymax'],
                    'zmin': bbox['zmin'],
                    'zmax': bbox['zmax'],
                    'surface_area': mprops['surface_area'],
                    'cg_x': mprops['gravity_center']['x'],
                    'cg_y': mprops['gravity_center']['y'],
                    'cg_z': mprops['gravity_center']['z'],
                    'density': mprops['density'],
                    'mass': mprops['mass'],
                    'volume': mprops['volume'],
                    'cg_inert_tensor_x': mprops['ctr_grav_inertia_tensor']['x_axis']['x'],
                    'cg_inert_tensor_y': mprops['ctr_grav_inertia_tensor']['y_axis']['y'],
                    'cg_inert_tensor_z': mprops['ctr_grav_inertia_tensor']['z_axis']['z'],
                    'HORZ_DIAMETER': hd,
                    'VERT_DIAMETER': vd,
                    'FUSE_CYL_LENGTH': tl,
                    'FLOOR_HEIGHT': fh
                }

                    #print(f"HORZ_DIAMETER: {hd}, VERT_DIAMETER: {vd}, TUBE_LENGTH: {tl}")
                writer.writerow(res)



            # for hd in range(780, 1020, 20):  # HORZ_DIAMETER
            #     for vd in range(70, 100, 20): # VERT_DIAMETER
            #         for tl in range(950, 1550, 50):  # TUBE_LENGTH
            #             for fh in range(3, int(vd/2), 10): 
            #                 set_property(creo_client, capsule, "HORZ_DIAMETER", hd)
            #                 set_property(creo_client, capsule, "VERT_DIAMETER", vd)
            #                 set_property(creo_client, capsule, "CYL_FUSE_LENGTH", tl)

            #                 regenerate(creo_client)
            #                 if with_tforms:
            #                     tforms = get_transform(creo_client, capsule)
            #                 bbox = get_bounding_box(creo_client, capsule)
            #                 mprops = get_massprops(creo_client, capsule)

            #                 bbox.update(mprops)

            #                 if with_tforms:
            #                     bbox.update(tforms)

            #                 #print(bbox)
            #                 res = {
            #                     'subject': subject,
            #                     'xmin': bbox['xmin'],
            #                     'xmax': bbox['xmax'],
            #                     'ymin': bbox['ymin'],
            #                     'ymax': bbox['ymax'],
            #                     'zmin': bbox['zmin'],
            #                     'zmax': bbox['zmax'],
            #                     'surface_area': mprops['surface_area'],
            #                     'cg_x': mprops['gravity_center']['x'],
            #                     'cg_y': mprops['gravity_center']['y'],
            #                     'cg_z': mprops['gravity_center']['z'],
            #                     'density': mprops['density'],
            #                     'mass': mprops['mass'],
            #                     'volume': mprops['volume'],
            #                     'cg_inert_tensor_x': mprops['ctr_grav_inertia_tensor']['x_axis']['x'],
            #                     'cg_inert_tensor_y': mprops['ctr_grav_inertia_tensor']['y_axis']['y'],
            #                     'cg_inert_tensor_z': mprops['ctr_grav_inertia_tensor']['z_axis']['z'],
            #                     'HORZ_DIAMETER': hd,
            #                     'VERT_DIAMETER': vd,
            #                     'TUBE_LENGTH': tl
            #                 }

            #                 print(f"HORZ_DIAMETER: {hd}, VERT_DIAMETER: {vd}, TUBE_LENGTH: {tl}")

            #                 writer.writerow(res)

def build_random_substructure():
    pass

if __name__ == "__main__":

    prt_path = "C:\\JWork\\Agents\\uam-uav-models\\CAD"
    uav_uam_models = os.listdir(prt_path)
    uav_uam_model_lookup = {pf.split(".")[0] : os.path.join(prt_path, pf) for pf in uav_uam_models}
    blank = prt_path + "\\blank.asm"
    capsule = uav_uam_model_lookup['fuse_capsule_new'].split(".")[0] + ".prt"
    tube = uav_uam_model_lookup['para_tube'].split(".")[0] + ".prt"
    cargo_case = uav_uam_model_lookup['para_cargo_case'].split(".")[0] + ".prt"
    orient = uav_uam_model_lookup['orientuav'].split(".")[0] + ".prt"
    uavwing = uav_uam_model_lookup['uav_wing'].split(".")[0] + ".prt"

    creo_client = Client(ip_adress="localhost", port=9056)
    creo_client.connect()
    
    frm = "PRT_CSYS_DEF"
    constraints = {'type':'csys','asmref': blank,'compref': frm}


    # # massprop collection for drag model
    for subject in ["capsule"]:
        if subject == "uavwing":
            prt_file = uavwing
        elif subject == "capsule":
            prt_file = capsule

        creo_client.file_open(
            prt_file,
            dirname=prt_path
        )
        collect_massprops(subject)


    