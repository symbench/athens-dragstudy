# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:08:02 2021

@author: fheim
"""

import json
import numpy as np
import trimesh
# from trimesh.viewer.windowed import SceneViewer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial.transform import Rotation as R
import time
import copy


plt.close('all')
start = time.time()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D([x, x + dx], [y, y + dy], [z, z + dz], *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

    
def searchlist(name, info, people):
    return [element['PROP_VALUE'] for element in people if element['COMP_NAME'] == name and element['PROP_NAME'] == info]

def isRectangleOverlap(R1, R2):
    if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
        return False
    else:
        return True


def interarea(R1, R2):  # returns None if rectangles don't intersect
    # Here we assume that the first corner is lower and second corner is upper
    dx = min(R1[2], R2[2]) - max(R1[0], R2[0])
    dy = min(R1[3], R2[3]) - max(R1[1], R2[1])
    A2 = (R2[2] - R2[0]) * (R2[3] - R2[1])
    if (dx >= 0) and (dy >= 0):
        return (dx * dy) / A2


def distlength(R1):
    L = np.sqrt((R1[1] - R1[0]) ** 2 + (R1[3] - R1[2]) ** 2 + (R1[5] - R1[4]) ** 2)  # x1,x2,y1,y2,z1,z2 is order
    return L


def charlength(R1):
    S = [abs((R1[2] - R1[0])), abs((R1[3] - R1[1]))]  # Side lengths of projected area rectangle
    AR = max(S) / min(S)  # Aspect ratio
    if AR > 3:  # Set threhold at 2
        L = min(S)  # If high aspect ratio use min side length
    else:  # If lower aspect, disk diameter approximation applies
        A = S[0] * S[1]  # Area of rectangle
        L = np.sqrt(A * 4 / np.pi)
    return L, AR

def flow_face(mf, mn):
    
    mn = np.round(mn, decimals=8)
    # morder = np.argsort(mf[:,1])[::-1]
    # mnormal = np.round(mn[morder,:], decimals=8)
    # mfacet = mf[morder,:]

    # leng = mfacet[np.argmax(mnormal[:, 1]), 1] - mfacet[np.argmin(mnormal[:, 1]), 1]
    
    # morder = np.argsort(mf[:,2])[::-1]
    # mnormal = np.round(mn[morder,:], decimals=8)
    # mfacet = mf[morder,:]
    
    
    # wid = mfacet[np.argmax(mnormal[:, 2]), 2] - mfacet[np.argmin(mnormal[:, 2]), 2]# mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[np.argmin(mesh.facets_normal[:, 2]), 2]

    # morder = np.argsort(mf[:,0])[::-1]
    # mnormal = np.round(mn[morder,:], decimals=8)
    # mfacet = mf[morder,:]
    
    # thi = mfacet[np.argmax(mnormal[:, 0]), 0] - mfacet[np.argmin(mnormal[:, 0]), 0]#mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[np.argmin(mesh.facets_normal[:, 0]), 0]
  
    leng = mf[np.argmax(mn[:, 1]), 1] - mf[np.argmin(mn[:, 1]), 1]
    if leng < 0.0001:
        leng = np.max(mf[:,1]) - np.min(mf[:,1])
        
    wid = mf[np.argmax(mn[:, 2]), 2] - mf[np.argmin(mn[:, 2]), 2]
    if wid < 0.0001:
        wid = np.max(mf[:,2]) - np.min(mf[:,2])
        
    thi = mf[np.argmax(mn[:, 0]), 0] - mf[np.argmin(mn[:, 0]), 0]
    if thi < 0.0001:
        thi = np.max(mf[:,0]) - np.min(mf[:,0])
  
    
    return(leng,wid,thi)
    
def modfun_disk(xd):
    # assumes interaction between two plates (8-1)
    # piecewise linear fits applied to experimental data seen in Figure 1.
    if xd < 2:
        # m = -0.1*xd
        m = 0.01
    elif xd > 7:  # 7
        m = 1
    else:
        m = 0.2 * xd - 0.4
        # m = 0.06*xd -0.4
    return m


def modfun_cyl(xd):
    # assumes interaction between two cylinders (8-1)
    # piecewise linear fits applied to experimental data seen in Figure 1.
    if xd < 2:  # very close
        # m = -0.1*xd
        m = 0.01
    elif xd > 20:  # very far 9
        m = 1
    else:
        m = 0.23  # 0.3/1.17
    return m


def cylinderdrag(cyl_len, cyl_dia, cyl_n, ang, vel, mu, rho):
    # For cylinder assumed to have axis normal to flow direction
    # Constants
    nu = mu / rho
    cyln = cyl_n.copy()
    cyln = np.abs(cyln)  # drag analysis is the same reguardless of -/+ orientation of trimesh found symmetry axis
    cyl_Re = np.transpose(np.tile(vel * cyl_dia / nu, [len(ang), 1]))

    cyln[np.abs(cyln) < 0.00001] = 0  # forget small angles
    # Get orginal pitch angle of cyl
    n = [0, 0, 1]
    angle = np.arctan2(np.cross(cyln, n), np.dot(cyln, n))
    # rangle = np.rad2deg(angle[0])
    pangle = np.rad2deg(angle[1])
    # yangle =np.rad2deg( angle[2])

    # pitch = np.tile(90-ang*np.cos(angle[0])+pangle,[1,1])#realtive to x axis like alpha (AoA) # before I understood that AOA cannot be negative in this formulation...
    pitch = np.tile(90 + ang * np.cos(angle[0]) + pangle, [len(vel), 1])

    # Approximations per laminar and turbulent flow (3-9)
    cyl_cd = np.ones(np.shape(cyl_Re)) * 0.3  # Assume turbulent
    keep = cyl_Re <= 550000
    cyl_cd[keep] = 1.18  # check if laminar

    # Correct for finite L/d(3-15)
    # Need if statement to avoid the shortcoming of the approximation
    if (cyl_dia / cyl_len) > 1:
        cyl_cd = cyl_cd * (1 - 0.5 * (1))
    else:
        cyl_cd = cyl_cd * (1 - 0.5 * (cyl_dia / cyl_len))

    # Correct for cross-flow(3-11) formulation is only for positive angles! so we need to make some adjustments...
    signs = np.sign(pitch) + (pitch == 0) * 1  # get sign, watches for the zero case
    pitch = abs(pitch)
    cyl_cl = cyl_cd * ((np.sin(np.deg2rad(pitch)) ** 2) * np.cos(np.deg2rad(pitch))) * signs
    cyl_cd = cyl_cd * (np.sin(np.deg2rad(pitch)) ** 3)

    # add face/axial contributions (Fig 21 (3-12))
    if cyl_len / cyl_dia > 1:

        #cyl_cd = cyl_cd + np.ones(cyl_cd.shape) * 0.8 * np.cos(np.deg2rad(pitch))
        cyl_cd = (cyl_cd * cyl_len * cyl_dia + np.ones(cyl_cd.shape) * 0.8 * np.cos(np.deg2rad(pitch)) * np.pi * cyl_dia**2/4) / (cyl_len * cyl_dia)
    else:

        #cyl_cd = cyl_cd + np.ones(cyl_cd.shape) * 1.2 * np.cos(np.deg2rad(pitch))
        cyl_cd = (cyl_cd * cyl_len * cyl_dia + np.ones(cyl_cd.shape) * 1.2 * np.cos(np.deg2rad(pitch)) * np.pi * cyl_dia**2/4) / (cyl_len * cyl_dia)
        
    # Add friction drag (2-5,2-7,3-11)
    cyl_cf = (1 / (3.46 * np.log(cyl_Re) - 5.6)) ** 2  # (2-5) Schoenherr emperical approximation for turbulent BL
    cyl_cf = cyl_cf + 0.0016 * (cyl_len / cyl_dia) / (
                cyl_Re ** (2 / 5))  # Correct for 3d cyldiner, assuming turbulent ((2-7, eq 32))
    cyl_cd = (cyl_cd * cyl_len * cyl_dia + np.pi * cyl_dia**2/4 * cyl_cf) / (
                cyl_len * cyl_dia)  # (3-11) diameter added 4/4/22 due to insight on page 2-8, normalized by refarea
    # Calculate reference area
    warea = cyl_len * cyl_dia

    return cyl_cd, cyl_cl, cyl_cf, warea


def platedrag(plate_len, plate_thi, plate_wid, plate_n, ang, vel, mu, rho, q):
    # Maybe this should be named box drag, as I assume it is a box (2 plates) and then overwrite info if it turns out to be a actual plate (single)
    # Here I am first computing the effect of a forward most face on a box. then I am computing the effet of the bottom and top faces. 
    # I get the normal of the front face which is why I do this one first
    # Make sure lengths are positive (6/6/2022)
    plate_len = abs(plate_len)
    plate_wid = abs(plate_wid)
    plate_thi = abs(plate_thi)

    ##6/6/22 I changed how this is being handled. First we consider the front most plate side, then we consider the lower plate for a box
    # span = plate_len, chord = plate_thi, and thickness = plate_wid per typical airfoil nomencalture
    # Here plate_len is the span, plate_wid is the height, and plate_thi is the chord, remember we are assuming its at an AOA of 90 deg, then modify from there.
    # For a plate with normal along the flow direction
    # Constants
    # ang = np.arange(-20,21,5)
    platen = plate_n.copy()
    nu = mu / rho
    plate_Re = np.transpose(np.tile(vel * plate_wid / nu, [len(ang), 1]))
    platen[np.abs(platen) < 0.00001] = 0

    # Get orginal pitch angle of plate
    n = [1, 0, 0]
    angle = np.arctan2(np.cross(n, platen), np.dot(n, platen))
    # rangle = angle[0]
    pangle = angle[1]
    # yangle = angle[2]
    pangle = np.rad2deg(pangle)
    # plate_aoa = np.tile(90+ang+pangle,[len(vel),1])#realtive to x axis like alpha (AoA)
    pitch0 = np.tile(ang + pangle, [len(vel), 1]) 
    pitch = np.tile(ang + pangle - 90, [len(vel), 1])  # realtive to x axis like alpha (AoA)
    signs0 = np.sign(pitch) + (pitch == 0) * 1
    pitch[abs(pitch) > 90] = abs(pitch[abs(pitch) > 90] + 180)
    signs = np.sign(pitch) + (pitch == 0) * 1  # get sign, watches for the zero case
    pitch = abs(pitch)

    ## Approximation per laminar and turbulent flow (3-15) These are for a plate at 90 deg AoA, but we then adjust for actual AOA
    # What I am doing here is: Assuming a 2d, infinite plate. Then there are 3 aspect ratio regions, where we adjust
    # plate_cd = np.ones(plate_Re.shape)*2 #(Eq. 29) Not used

    # #Correct for aspect ratio
    # if plate_len/plate_wid<5:
    #     plate_cd = np.ones(plate_Re.shape)*1.18
    # elif plate_len/plate_wid>=5 and plate_len/plate_wid<=15:
    #     plate_cd = np.ones(plate_Re.shape)*1.3
    # else:
    #     plate_cd = plate_cd*(1-5*(plate_wid/plate_len)) #(3-16, explains this)

    # 4/1/22 remove cross flow, which is really for cylinders and tackle the plate at angle problem
    # #Correct for cross-flow(3-11) formulation is only for positive angles! so we need to make some adjustments...
    # signs = np.sign(plate_aoa) + (plate_aoa)*1# get sign, watches for the zero case
    # plate_cl = plate_cd*((np.sin(np.deg2rad(plate_aoa))**2)*np.cos(np.deg2rad(plate_aoa)))*signs
    # plate_cd = plate_cd*(np.sin(np.deg2rad(plate_aoa))**3)*signs

    # # plates at pitch angles, evidently due to a lack of suction, do not have an induced and angle of attack, which makes this simpler
    # #(3-16) has Figure 29a, which presents results for a square disk. I will add this for a comparision to the small aspect ratio formulation. Aspect ratio is span/chord, which at 90 deg is len/wid 
    # if plate_len / plate_wid >= 0.8 and plate_len / plate_wid <= 1.2: # we are extrapolating slightly, as it should be close
    #     #Plate aerodynamic force
    #     plate_cn = np.ones(plate_Re.shape) * 1.17 # This is true between AOA 45 to 90 deg
    #     here = plate_aoa < 45
    #     plate_cn[here] = (1.8/45) * plate_aoa[here] # modify those at less than 45 deg

    #### from pressure drag section
    # plates at pitch angles, evidently due to a lack of suction, do not have an induced and angle of attack, which makes this simpler
    # (3-16) has Figure 29a, which presents results for a square disk. I will add the aspect ratio regions from (3-16). Aspect ratio is span/chord, which at 90 deg is len/thi for front plate
    if plate_len / plate_wid < 5:  # we are extrapolating slightly, as it should be close
        # Plate aerodynamic force
        plate_cn = np.ones(plate_Re.shape) * 1.18  # These are true between AOA 45 to 90 deg
    elif plate_len / plate_wid >= 5 and plate_len / plate_wid <= 15:
        plate_cn = np.ones(plate_Re.shape) * 1.3  # These are true between AOA 45 to 90 deg
    else:
        plate_cn = 2 * np.ones(plate_Re.shape) * (
                    1 - 5 * (plate_wid / plate_len))  # (3-16, explains this)# These are true between AOA 45 to 90 deg
    # Then we do the correction for AOA
    here = pitch < 45
    plate_cn[here] = (1.8 / 45) * pitch[here]  # modify those at less than 45 deg
    plate_cl = plate_cn * np.cos(
        np.deg2rad(pitch)) * signs  # we assume plate effect is symmetric and just add the sign back in
    plate_cd = plate_cn * np.sin(np.deg2rad(pitch)) 

    #  #### from drag from lift section   
    # if plate_len / plate_wid > 2:
    #     plate_cl = 2*np.pi*np.sin(np.deg2rad(pitch)) #Thin airfoil theory
    #     plate_cd =  plate_cl * np.tan(np.deg2rad(pitch)) + (1/(2*np.pi)) # (7-3) base drag + induced drag + flow reattachment contribution due to sharp edges at AOA, displayed in Figure 4
    # else:

    #     plate_clo = 0.5* np.pi * plate_len  / plate_wid * np.sin (np.deg2rad(pitch)) #(7-16), Eq. 30 # This is from line lifting theory
    #     plate_dcl_da = 1/(2 / np.pi * (plate_len / plate_wid)) # (7-17) Eq. 33 # this is for flat plates
    #     plate_cd = plate_clo*np.tan(np.deg2rad(pitch)) + plate_dcl_da* np.tan(np.deg2rad(pitch)) #(7-18) Eq. 37
    #     plate_cl = plate_clo + plate_dcl_da*np.deg2rad(pitch) # add circulation contributuion and vortex lattice contribution (7-18, around Eq. 36)
    #   ####  

    # Side edges, formulation is for pairs of sides
    # plate_cf = plate_cf + (2.9*(plate_thi/plate_wid)/plate_Re) / (0.03*(plate_thi/plate_wid)/(plate_Re**(1/5))) # Here we add laminar BL 3d effects then the final division is for turbulent correction (2-8, eq. 33 & 34)
    # # top and bottom edges
    # plate_cf = plate_cf + (2.9*(plate_thi/plate_len)/plate_Re) / (0.03*(plate_thi/plate_len)/(plate_Re**(1/5)))

    # Add friction drag (2-5,2-8, 3-11) of front plate
    plate_cf = (1 / (3.46 * np.log(plate_Re) - 5.6)) ** 2  # (2-5) Schoenherr emperical approximation for turbulent BL

    # top and bottom edges of plate
    plate_cf = plate_cf + (0.03 * (plate_thi / plate_len) / (plate_Re ** (1 / 5)))
    # front face things, corrected 8/8/22
    plate_cf = plate_cf + (0.03 * (plate_wid / plate_len) / (
                plate_Re ** (1 / 5)))  # Here we add 3D turbulent correction (2-8, eq. 34)
    # Here we check to see if this is a plate or a box. If it is a box we add the other sides
    box_cd = 0
    box_cl = 0
    if plate_wid / plate_thi < 1 / 10:  # If its a plate then we forget what we just computed. Keep friction though
        plate_cd = plate_cd * 0
        plate_cl = plate_cl * 0

    #### Now we do the plate like consideration relative to zero AoA
    # For a plate with normal along the flow direction
    pitch = pitch0#pitch * signs + 90  # realtive to x axis like alpha (AoA)
    pitch[abs(pitch) > 90] = (abs(pitch[abs(pitch) > 90]) - 180) * signs0[abs(pitch) > 90]
    signs = np.sign(pitch0 ) + (pitch0 == 0) * 1  # get sign, watches for the zero case
    pitch = abs(pitch)
    # Just repeat but with thi as chord (replaces width)
    box_Re = np.transpose(np.tile(vel * plate_thi / nu, [len(ang), 1]))
    if plate_len / plate_thi < 5:  # we are extrapolating slightly, as it should be close
        # Plate aerodynamic force
        box_cn = np.ones(box_Re.shape) * 1.18  # These are true between AOA 45 to 90 deg
    elif plate_len / plate_thi >= 5 and plate_len / plate_thi <= 15:
        box_cn = np.ones(box_Re.shape) * 1.3  # These are true between AOA 45 to 90 deg
    else:
        box_cn = 2 * np.ones(box_Re.shape) * (
                    1 - 5 * (plate_thi / plate_len))  # (3-16, explains this)# These are true between AOA 45 to 90 deg
    # Then we do the correction for AOA
    here = pitch < 45
    box_cn[here] = (1.8 / 45) * pitch[here]  # modify those at less than 45 deg
    box_cl = box_cn * np.cos(np.deg2rad(pitch)) * signs
    box_cd = box_cn * np.sin(np.deg2rad(pitch)) 

    # Add front face contribution if box    
    plate_cd = (plate_cd * plate_len * plate_wid + box_cd * plate_len * plate_thi) / (
                plate_len * plate_thi)  # here we are normalizing by the bottom plate area, not the forward facing one
    plate_cl = (plate_cl * plate_len * plate_wid + box_cl * plate_len * plate_thi) / (plate_len * plate_thi)
    plate_cd = (plate_cd * plate_len * plate_thi + 2 * plate_len * plate_thi * plate_cf) / (
                plate_len * plate_thi)  # (3-11) plate insight on (2-8) b is the span I believe 4/4/22. normalize by refarea
    # Calculate reference area
    warea = plate_len * plate_thi  # always will hold true reguardless of orginal orientation, equivalent to planform area
    # print(plate_len*1000)
    # print(plate_wid*1000)
    # print(plate_thi*1000)
    return plate_cd, plate_cl, plate_cf, warea


def ellipticaldrag(ell_chord, ell_len, ell_dia, ell_n, ang, vel, mu, rho):
    # For a streamline, elliptical, symmetrical body we assume flow is parallel to the larger axis length, according to Fig. 22 on (6-16)
    nu = mu / rho
    elln = ell_n.copy()
    ell_Re = np.transpose(np.tile(vel * ell_chord / nu, [len(ang), 1]))

    elln[np.abs(elln) < 0.00001] = 0  # forget small angles
    # Get orginal pitch angle of ellipsoid
    n = [1, 0, 0]
    angle = np.arctan2(np.cross(elln, n), np.dot(elln, n))
    # rangle = np.rad2deg(angle[0])
    pangle = np.rad2deg(angle[1])
    yangle = np.rad2deg(angle[2])
    pitch = np.tile(ang + pangle, [len(vel), 1])  #
    # 6/8/22 Add something to consider if the fuselage is rotated for the 3d analysis
    if abs(yangle) > abs(pangle) or abs(
            pangle) > 60:  # if we have a large yaw angle or we have a large pitch angle, which invalidates some assumptions. Therefore, we adjust if so.
        # ell_len and ell_chord become equal, ell_dia becomes ell_chord
        ell_chord = ell_dia
        ell_dia = ell_len
        ell_len = ell_chord  # which is the diameter
        if abs(pangle) > 60:  # addtionally
            pitch = np.tile(0, [len(vel), 1])  # cylinder has no pitch angle so set to zero

    ## (6-18) pressure drag follows the trend of skin-friction drag for a streamline body. I looked at the 3 models and selcted one
    ## Formulation (3-11) for ellipsoid is used below
    # Skin friction drag approximation
    # Add friction drag (2-5,2-7,3-11) #note this is a spot on approximation for the ellipsoid, according to Figure 19
    ell_cf = (1 / (3.46 * np.log(
        ell_Re) - 5.6)) ** 2  # (2-5) Schoenherr emperical approximation for turbulent BL. There does not appear to be a need to modify this for streamline bodies

    # There is some information about 2D elliptical drag
    # ell_cd = ell_cf * (4 + 2*(ell_chord/ell_dia) + 120 * (ell_dia/ell_chord)**2)(3-11, Eq. 21) 2D section Turbulent
    # ell_cd = ell_cf * 2 * (1 + ell_chord/ell_dia) + 1.1 * ell_dia/ell_chord #(3-11, Eq. 20) 2D section Laminar

    # Check to see the Reynolds number to determine if laminar or turbulent flow
    # 3D Ellipsoidal drag, assume laminar
    ell_cd = 0.44 * ell_dia / ell_len + 4 * ell_cf * (ell_len / ell_dia) + 4 * ell_cf * (
                ell_dia / ell_len) ** 0.5  # (3-12) 3D ellipsoids # It appears that this drag does not change much with angle of attack
    keep = ell_Re >= 10 ** 6  # if above transtion Reynolds number, then we make it turbulent, fixed 8/8/22
    # Pressure Drag turbulent approximation per wetted area estimation Eq. 31 on (6-18) for 3D body
    ell_cd[keep] = 0.004 * (3 * (ell_chord / ell_dia) + 4.5 * (ell_dia / ell_chord) ** 0.5 + 21 * (
                ell_dia / ell_chord) ** 2)  # (6-18, Eq. 31) Streamline bodies. 3-12, says fit well for turbulent if cf = 0.004, hence the hardcoded value

    ## Ellipsoid lift
    # (7-20) states that most drag is parasitic and very little is induced by lift of a streamline body.
    # (7-19)states that streamline bodies do not exhibit much crossflow. This maybe true due to most pilots keeping fuselages pointed "into the wind

    # Use eq 31 on page (7-16)
    ell_cl = 0.5 * np.pi * np.sin(np.deg2rad(pitch)) * 0.438# correct for smooth round (0.012 / 0.0274) = 0.438 from Figure 33 (7-19)

    # Account for potential trailing edge truncation (3-22)
    # (3-22) Fig. 41 depicts truncation added drag. Fig. 42 indicates some benefit to lift from tuncating >40 % thickness profiles
    if ell_dia / ell_chord > 0.4:  # If t/c is greater than 40 % we see a benefit to truncation, else its a detriment
        # Correct for cross-flow(3-11) Here we add the lift increase and drag decrease, discussed in (3-22). The advantage increases with t/c but only slightly, so we will ignore that and fix to 20%.
        ell_cl = ell_cl * 1.2  # Increase Cl by 20%
        ell_cd = ell_cd * 0.8  # decrease Cd by 20%
    else:
        ell_cd = ell_cd + (0.34 / (ell_cd) ** (1 / 3)) * ((ell_chord - ell_len) / ell_chord) ** (
                    4 / 3)  # cd_delta Eq. 42 on (3-22)
    

    # Combine friction and pressure drag and drag due to lift
    ell_cd = ell_cd + ell_cf + ell_cl**2 / np.pi # (3-11)
    
    # Calculate reference area
    warea = np.pi * (ell_dia / 2) ** 2  # This differs from cylinder because, here, we assume long axis parallel to flow
    return ell_cd, ell_cl, ell_cf, warea


#############################################################################################################################################################################################################################

def run_full(DataName, ParaName, include_wing, create_plot, debug, stl_output, struct):
    vp = 1#5  # 30 m/s
    ap = 1#4  # +0 deg
    mp = 100  # 50#scale of arrows? 50 for UAM, 100 for UAV works well

    vel = np.arange(15,31,15)#np.arange(5, 51, 5)  # Range of velocties for table
    mu = 18 * 10 ** -6  # air viscosity
    rho = 1.225  # air density
    ang = np.arange(-20, 1, 20)# np.arange(-20, 21, 50)
    Vel = np.transpose(np.tile(vel, [len(ang), 1]))

    print(f"type DataName {isinstance(DataName, dict)}")
    print(f"type ParaName {isinstance(ParaName, dict)}")
    
    if isinstance(DataName, str) and isinstance(ParaName, str):
        with open(DataName) as f:
            spatial = json.load(f)
        with open(ParaName) as f:
            parameter = json.load(f)
    elif isinstance(DataName, dict) and isinstance(ParaName, dict):
        print("assignment")
        spatial = DataName
        parameter = ParaName
    
    # with open(ConnName) as f:
    #     connect = json.load(f)
    # with open(PropName) as f:
    #     properties = json.load(f)
  
    if stl_output == True :
        Mesh_list = list()
        
    if struct == True:
        structure = copy.deepcopy(parameter)
    prop={}    
    #####Propeller Interference Section
    #Do propellers first, assume rigid body and rigid wake of D*2, we need the normals
    for q in spatial:
        if "prop" in parameter[q]['CADPART']:
            # print(q)
            Trans = np.array(spatial[q]["Translation"])
            CG = np.array(spatial[q]["CG"])
            Trans = Trans.astype(float)
            CG = CG.astype(float)
            CG = CG + Trans
            Trans = np.vstack(Trans)
            Rot = np.array(spatial[q]["Rotation"])
            Rot = Rot.astype(float)

            mesh2cad = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, float(parameter[q]["HUB_THICKNESS"]) / 2], [0, 0, 0, 1]]
            Tform = np.hstack((Rot, Trans))
            Tform = np.vstack((Tform, (0, 0, 0, 1)))

            # prop mesh with diamter=1 for interference projection, according to Fluid-Dynamic Lift by Hoerner on 12-9. This ratio varies from .92 to .816. Here we will assume 1
            mesh = trimesh.creation.cylinder(float(parameter[q]["DIAMETER"])/2, float(parameter[q]["HUB_THICKNESS"]),
                                              sections=20, segment=None, transform=mesh2cad,
                                              center_mass=spatial[q]["CG"])
            mesh.apply_transform(Tform)
            
            prop[q] = copy.deepcopy(spatial[q])
            prop[q]["N"] = mesh.symmetry_axis
            prop[q]['Poly'] = trimesh.path.polygons.projected(mesh,mesh.symmetry_axis) # projected polygon
            prop[q]['polyp']=list()
            prop[q]["GCG"] = mesh.center_mass
            prop[q]["Diameter"] = float(parameter[q]["DIAMETER"])
            prop[q]["Blockage_Area"] = prop[q]['Poly'].difference(prop[q]['Poly'])
            prop[q]["CdS"] = 0
            prop[q]["Part_Int"] = list()
            del mesh
    ####        
            
    for o in np.arange(1,4, 1):  # change second digit to 4  to get all 3 directions
        if create_plot == True:
            fig = plt.figure(figsize=plt.figaspect(1))
            ax = fig.add_subplot(111, projection='3d')

        ## conditional to only get structural output from forward flight case   
        if struct==True and o==1:
            struct = False
            structo=True
        elif struct==False and o==1:
            structo = False
        elif struct==False:
            
            structo=False
        ##Main part loop
        for q in spatial:
            Trans = np.array(spatial[q]["Translation"])
            CG = np.array(spatial[q]["CG"])
            Trans = Trans.astype(float)
            CG = CG.astype(float)
            CG = CG + Trans
            Trans = np.vstack(Trans)
            Rot = np.array(spatial[q]["Rotation"])
            Rot = Rot.astype(float)
            #Rot=Rot.transpose()

            Tform = np.hstack((Rot, Trans))
          
            Tform = np.vstack((Tform, (0, 0, 0, 1)))
            if o == 1:
                rotr = trimesh.transformations.rotation_matrix(np.deg2rad(0), [1, 0, 0])  # should be -90
                Tform = trimesh.transformations.concatenate_matrices(rotr, Tform)
             
            elif o == 2:
                rotr = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [0, 0, 1])  # should be -90
                Tform = trimesh.transformations.concatenate_matrices(rotr, Tform)
            elif o == 3:
                rotr = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0])  # should be -90
                Tform = trimesh.transformations.concatenate_matrices(rotr, Tform)
              
            if "motor" in parameter[q]['CADPART']:
                # Flip y and z axis
                mesh2cad = [[1, 0, 0, 0],
                            [0, np.cos(np.pi / 2), -np.sin(np.pi / 2), float(parameter[q]["LENGTH"]) * 1.55 / 2],
                            [0, np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 0, 1]]
                
              

                mesh = trimesh.creation.cylinder(float(parameter[q]["CAN_DIAMETER"]) / 2,
                                                 float(parameter[q]["LENGTH"]) * 1.5, sections=10, segment=None,
                                                 transform=mesh2cad, center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)

                cd, cl, cf, warea = cylinderdrag((float(parameter[q]["LENGTH"]) * 1.5) / 1000,
                                                 (float(parameter[q]["CAN_DIAMETER"])) / 1000, mesh.symmetry_axis, ang,
                                                 vel, mu, rho)

                ecolor = [177.3 / 255, 189.4 / 255, 201.5 / 255]

            elif "esc" in parameter[q]['CADPART']:
                mesh2cad = [[1, 0, 0, (float(parameter[q]["LENGTH"]) / 2) - 15],
                            [0, 1, 0, float(parameter[q]["THICKNESS"]) / 2],
                            [0, 0, 1, -float(parameter[q]["WIDTH"]) / 2], [0, 0, 0, 1]]
                mesh = trimesh.creation.box(extents=(
                float(parameter[q]["LENGTH"]), float(parameter[q]["THICKNESS"]), float(parameter[q]["WIDTH"])),
                                            transform=mesh2cad, center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)

                front_index = np.argmax(mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
                leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
                # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 1]), 1]
                # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 2]), 2]
                # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 0]), 0]
                cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
                                              ang, vel, mu, rho, q)

                ecolor = [2.8 / 255, 3.1 / 255, 255 / 255]

            elif "tube" in parameter[q]['CADPART']:
                mesh2cad = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, float(parameter[q]["LENGTH"]) / 2], [0, 0, 0, 1]]
                #mesh2cad = [[1, 0, 0, 0], [0, np.cos(np.pi/2), -np.sin(np.pi/2), 0], [0, np.sin(np.pi/2), np.cos(np.pi/2), float(parameter[q]["Length"]) / 2], [0, 0, 0, 1]]

                # mesh = trimesh.creation.cylinder(float(parameter[q]["OD"]) / 2, float(parameter[q]["LENGTH"]),
                #                                  sections=10, segment=None, transform=mesh2cad,
                #                                  center_mass=spatial[q]["CG"])
                # mesh.apply_transform(Tform)
                # cd, cl, cf, warea = cylinderdrag((float(parameter[q]["LENGTH"])) / 1000,
                #                                  (float(parameter[q]["OD"])) / 1000, mesh.symmetry_axis, ang, vel, mu,
                #                                  rho)
                
                #mesh2cad = [[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0], [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
        
                
                mesh = trimesh.creation.cylinder(float(parameter[q]["OD"]) / 2, float(parameter[q]["LENGTH"]),
                                                 sections=10, segment=None, transform=mesh2cad,
                                                 center_mass=spatial[q]["CG"])
              
                
                mesh.apply_transform(Tform)
                
                cd, cl, cf, warea = cylinderdrag((float(parameter[q]["LENGTH"])) / 1000,
                                                 (float(parameter[q]["OD"])) / 1000, mesh.symmetry_axis, ang, vel, mu,rho)

                ecolor = [76.5 / 255, 73.9 / 255, 76.5 / 255]
                ## structure
                if struct==True:  
                    structure[q]["E"] = 228e9#Pa
                    structure[q]["G"] = 12e9#Pa
                    structure[q]["type"] = "circular"
                    
                    structure[q]["normal"] = np.round(mesh.symmetry_axis,10)
                    
                    norm = structure[q]["normal"]  * mesh.bounds
                    cent = (mesh.center_mass - mesh.center_mass * structure[q]["normal"])
                  # ### put temporary 6/30/22
                  #   structure[q]["MASS"] = (np.pi*float(structure[q]["LENGTH"]) * (float(structure[q]["OD"])**2 - float(structure[q]["ID"])**2)/4)* 0.0015500747 / 1000# densituy in g/mm^3 so ajust
                  
                  ###
                    structure[q]["p1"] = cent+norm[0]
                    structure[q]["p2"] = cent+norm[1]
                    structure[q]["p3"] = structure[q]["p1"] + np.cross(np.diff(mesh.bounds, axis = 0),structure[q]["normal"])/2#you have the normal figure out how to apply
                    # structure[q]["I"] = [[1/12 * float(structure[q]["MASS"]) * (3*((float(structure[q]["OD"])/2)**2 + (float(structure[q]["ID"])/2)**2) + float(structure[q]["LENGTH"])**2), 0,0],
                    #                   [0,1/12 * float(structure[q]["MASS"]) * (3*((float(structure[q]["OD"])/2)**2 + (float(structure[q]["ID"])/2)**2) + float(structure[q]["LENGTH"])**2), 0],
                    #                   [0,0,1/2*float(structure[q]["MASS"])*((float(structure[q]["OD"])/2)**2 + (float(structure[q]["ID"])/2)**2)]]# defined according to trimesh cylinder creation coordinates
                    structure[q]["I"] = [[ np.pi/4 * ((float(structure[q]["OD"])/2)**4 + (float(structure[q]["ID"])/2)**4), 0,0],
                                      [0,np.pi/4 * ((float(structure[q]["OD"])/2)**4 + (float(structure[q]["ID"])/2)**4), 0],
                                      [0,0,np.pi/2 * ((float(structure[q]["OD"])/2)**4 + (float(structure[q]["ID"])/2)**4)]] # assume z points along length, per trimesh
                    
                    structure[q]["I"] = trimesh.inertia.transform_inertia(mesh2cad, structure[q]["I"])# Get into the part coordinate system
                    structure[q]["I"] = trimesh.inertia.transform_inertia(Tform, structure[q]["I"])# Get into the assembly coordinate system
                    
                    

            elif "flange" in parameter[q]['CADPART']:
                mesh2cad = [[1, 0, 0, 0], [0, 1, 0, float(parameter[q]["BOX"]) * 1.1 / 2], [0, 0, 1, 0], [0, 0, 0, 1]]

                mesh = trimesh.creation.box(extents=(
                float(parameter[q]["BOX"]) / 2, float(parameter[q]["BOX"]) * 1.1, float(parameter[q]["BOX"]) / 2),
                                            transform=mesh2cad, center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)

                front_index = np.argmax(mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
                leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
                
                # fig = plt.figure()
                # ax1 = fig.add_subplot(projection='3d')
                # ax1.scatter(mesh.facets_origin[:,0],mesh.facets_origin[:,1],mesh.facets_origin[:,2])
                # ax1.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces, Z=mesh.vertices[:, 2],
                #                 color=(0, 0, 0, 0), edgecolor=ecolor)
                cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
                                              ang, vel, mu, rho,q)
                ecolor = [219.3 / 255, 147.9 / 255, 0]
                ## structure
                if struct==True:  
                    structure[q]["E"] = 0.205e9#Pa 3d printed ABS, ref: https://mae.ufl.edu/rapidpro/pages/3D%20Printing%20Paper%20Final%20Manuscript.pdf
                    structure[q]["G"] = 0.6e9#Pa
                    structure[q]["type"] = "connection"
                    structure[q]["p1"] = mesh.center_mass

            elif "plate" in parameter[q]['CADPART']:
                mesh2cad = [[1, 0, 0, 0], [0, 1, 0, float(parameter[q]["THICKNESS"]) / 2], [0, 0, 1, 0], [0, 0, 0, 1]]
                mesh = trimesh.creation.box(extents=(
                float(parameter[q]["WIDTH"]), float(parameter[q]["THICKNESS"]), float(parameter[q]["LENGTH"])),
                                            transform=mesh2cad, center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)

                # Determine front face
                front_index = np.argmax(mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
                # which_face_par_to_flow = np.argmin(abs(mesh.facets_normal[front_index,]))# which direction is paralle to flow (dimension ignored in plate)
                leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
                # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 1]), 1]
                # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 2]), 2]
                # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 0]), 0]
                cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
                                              ang, vel, mu, rho,q)

                ecolor = [0, 0, 0]
                

                 
                    
            elif "fuse_capsule_new" in parameter[q]['CADPART']:
                
                diafuse = (float(parameter[q]["HORZ_DIAMETER"]) + float(parameter[q]["VERT_DIAMETER"])) /2 # Take average diameter
                # mesh2cad = [[np.cos(np.pi / 2), 0, np.sin(np.pi / 2), 0], [0, 1, 0, 0],
                #             [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2), 0], [0, 0, 0, 1]]
                mesh2cad = [[np.cos(3*np.pi/2), 0, np.sin(3*np.pi/2), 0], [0, 1, 0, 0],
                            [-np.sin(3*np.pi/2), 0, np.cos(3*np.pi/2), 0], [0, 0, 0, 1]]
                # First we create a cylinder to get orientation

                mesh = trimesh.creation.cylinder(diafuse / 2, height=float(parameter[q]["TUBE_LENGTH"]),
                                                 transform=mesh2cad, center_mass=spatial[q]["CG"], sections=10)

                mesh.apply_transform(Tform)
                cap_norm = mesh.symmetry_axis
                cap_cg = mesh.center_mass
                mesh = trimesh.creation.capsule(height=float(parameter[q]["TUBE_LENGTH"]), radius=diafuse / 2,
                                                count=[10, 10])

                # capsule is strange and won't let me add these in initial step
                mesh.apply_transform(mesh2cad)
                mesh.apply_transform(Tform)

                mesh.center_mass = cap_cg

                ecolor = [162.3 / 255, 174.7 / 255, 238.9 / 255]

                cd, cl, cf, warea = ellipticaldrag((float(parameter[q]["TUBE_LENGTH"]) + diafuse) / 1000,
                                                   (float(parameter[q]["TUBE_LENGTH"]) + diafuse) / 1000, diafuse / 1000, cap_norm, ang,
                                                   vel, mu, rho)
                
            ## structure
                if struct==True:
                    structure[q]["E"] = 71.7e9#Pa ref, https://www.matweb.com/search/DataSheet.aspx?MatGUID=4f19a42be94546b686bbf43f79c51b7d&ckck=1
                    structure[q]["G"] = 26.9e9#Pa
                    structure[q]["type"] = "fuselage"
                    structure[q]["p1"] = mesh.center_mass
                    
            elif "para_cargo" == parameter[q]['CADPART']:
                mesh2cad = [[1, 0, 0, 0], [0, 1, 0,  0], [0, 0, 1, float(parameter[q]["HEIGHT"])/2], [0, 0, 0, 1]]
                mesh = trimesh.creation.box(extents=(
                float(parameter[q]["LENGTH"]), float(parameter[q]["WIDTH"]), float(parameter[q]["HEIGHT"])),
                                            transform=mesh2cad, center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)

                # Determine front face
                front_index = np.argmax(mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
                # which_face_par_to_flow = np.argmin(abs(mesh.facets_normal[front_index,]))# which direction is paralle to flow (dimension ignored in plate)
                leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
                # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 1]), 1]
                # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 2]), 2]
                # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 0]), 0]
                cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
                                              ang, vel, mu, rho,q)

                ecolor = [200 / 255, 200 / 255, 200 / 255]
                
         
            elif "uav_wing" in parameter[q]['CADPART']:
                # Create polygon for potentially tapered wing.
                cs = [float(parameter[q]["CHORD_2"]),float(parameter[q]["CHORD_1"])]
                csi = (cs==np.min(cs))*1
                offset = np.max(cs)* float(parameter[q]["TAPER_OFFSET"]) - np.min(cs) * float(parameter[q]["TAPER_OFFSET"])
                wingperi = trimesh.path.polygons.edges_to_polygons(np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), np.array(
                    [[offset * csi[1], 0], [offset * csi[0], float(parameter[q]["SPAN"])],
                     [float(parameter[q]["CHORD_2"]) + offset * csi[0], float(parameter[q]["SPAN"])],
                     [float(parameter[q]["CHORD_1"])+ offset * csi[1], 0]]))  # For some reason this comes out as a list so choose [0]
                
                wingthic = np.max([float(parameter[q]["CHORD_2"]), float(parameter[q]["CHORD_1"])]) * float(
                    parameter[q]["THICKNESS"]) / 100  # Assume wing to be at max thickness
                # Rotate about x axis
                mesh2cad = [[1, 0, 0, 0], [0, np.cos(np.pi / 2), -np.sin(np.pi / 2), wingthic / 2],
                            [0, np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 0, 1]]

                mesh = trimesh.creation.extrude_polygon(wingperi[0], wingthic, transform=mesh2cad,
                                                        center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)
                ecolor = [128.7 / 255, 3.9 / 255, 3.9 / 255]
                if include_wing == True:
                    front_index = np.argmax(
                        mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
                    leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
                    # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
                    #     np.argmin(mesh.facets_normal[:, 1]), 1]
                    # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
                    #     np.argmin(mesh.facets_normal[:, 2]), 2]
                    # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
                    #     np.argmin(mesh.facets_normal[:, 0]), 0]
                    cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
                                                  ang, vel, mu, rho,q)
                    del leng, wid, thi, front_index
                else:
                        spatial[q]["GCG"] = mesh.center_mass
            ## structure
                if struct==True:
                    structure[q]["E"] = 228e9#Pa
                    structure[q]["G"] = 12e9#Pa
                    structure[q]["type"] = "wing"
                    airfoil_area = float(parameter[q]["THICKNESS"])/ 100 * ((float(parameter[q]["CHORD_1"]) + float(parameter[q]["CHORD_2"])) / 2)**2 # span times mean chord
                    structure[q]["normal"] = np.round(mesh.facets_normal[np.argmin(mesh.facets_area - airfoil_area)] ,10) # Points down span
                    
                    norm = structure[q]["normal"]  * mesh.bounds
                    cent = (mesh.center_mass - mesh.center_mass * structure[q]["normal"])
                  ###
                    structure[q]["p1"] = cent+norm[0]
                    structure[q]["p2"] = cent+norm[1]
                    structure[q]["p3"] = structure[q]["p1"] + np.cross(np.diff(mesh.bounds, axis = 0),structure[q]["normal"])/2#you have the normal figure out how to apply
                    # for structural purposes we assume a square spar with length = airfoil thickness
                    bh = float(parameter[q]["THICKNESS"])/ 100 * ((float(parameter[q]["CHORD_1"]) + float(parameter[q]["CHORD_2"])) / 2)
                    structure[q]["I"] = [[bh**4 / 12, 0,0],
                                      [0,bh**4 / 12, 0],
                                      [0,0,0]] # assume z points along length, per trimesh
                    
                    structure[q]["I"] = trimesh.inertia.transform_inertia(mesh2cad, structure[q]["I"])# Get into the part coordinate system
                    structure[q]["I"] = trimesh.inertia.transform_inertia(Tform, structure[q]["I"])# Get into the assembly coordinate system
                    

            elif "prop" in parameter[q]['CADPART']:
                mesh2cad = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, float(parameter[q]["HUB_THICKNESS"]) / 2],
                            [0, 0, 0, 1]]
                mesh = trimesh.creation.cylinder(float(parameter[q]["DIAMETER"]) / 2,
                                                 float(parameter[q]["HUB_THICKNESS"]), sections=15, segment=None,
                                                 transform=mesh2cad, center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)
                ecolor = [200 / 255, 200 / 255, 200 / 255]
                spatial[q]["GCG"] = mesh.center_mass
            
            elif "_hub_" in parameter[q]['CADPART']:
                mesh2cad = [[1,0,0,0],[0,1,0,0], [0,0,1,0], [0,0,0,1]]
                mesh = trimesh.creation.box(extents=(66, 20, 66),transform = mesh2cad, center_mass = spatial[q]["CG"])
                mesh.apply_transform(Tform)
                ecolor = [2.8/255, 3.1/255,255/255]
                spatial[q]["GCG"] = mesh.center_mass
                ##structure
                if struct==True:   
                    structure[q]["E"] = 0.205e9#Pa 3d printed ABS, ref: https://mae.ufl.edu/rapidpro/pages/3D%20Printing%20Paper%20Final%20Manuscript.pdf
                    structure[q]["G"] = 0.6e9#Pa
                    structure[q]["type"] = "connection"
                    structure[q]["p1"] = mesh.center_mass
            #7/8/22  batteries  should be in a fuselage now!
            elif "para_battery" in parameter[q]['CADPART']:
                mesh2cad = [[1, 0, 0, 0], [0, 1, 0, float(parameter[q]["WIDTH"]) / 2], [0, 0, 1, 0], [0, 0, 0, 1]]
                mesh = trimesh.creation.box(extents=(
                float(parameter[q]["THICKNESS"]), float(parameter[q]["WIDTH"]), float(parameter[q]["LENGTH"])),
                                            transform=mesh2cad, center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)
                spatial[q]["GCG"] = mesh.center_mass

                # front_index = np.argmax(mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
                # leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
                # # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
                # #     np.argmin(mesh.facets_normal[:, 1]), 1]
                # # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
                # #     np.argmin(mesh.facets_normal[:, 2]), 2]
                # # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
                # #     np.argmin(mesh.facets_normal[:, 0]), 0]
                # cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
                #                               ang, vel, mu, rho,q)
                ecolor = [255 / 255, 255 / 255, 0]
            # elif "wing_right" in parameter[q]['CADPART']:
            #     mesh2cad = [[1, 0, 0, -float(parameter[q]["CHORD"]) / 2], [0, 1, 0, float(parameter[q]["SPAN"]) / 4],
            #                 [0, 0, 1, 0], [0, 0, 0, 1]]
            #     mesh = trimesh.creation.box(extents=(
            #     float(parameter[q]["CHORD"]), float(parameter[q]["SPAN"]) / 2, float(parameter[q]["LASTTWO"])),
            #                                 transform=mesh2cad, center_mass=spatial[q]["CG"])
            #     mesh.apply_transform(Tform)
            #     ecolor = [255 / 255, 0 / 255, 0 / 255]
            #     if include_wing == True:
            #         front_index = np.argmax(
            #             mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
            #         leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
            #         # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
            #         #     np.argmin(mesh.facets_normal[:, 1]), 1]
            #         # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
            #         #     np.argmin(mesh.facets_normal[:, 2]), 2]
            #         # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
            #         #     np.argmin(mesh.facets_normal[:, 0]), 0]
            #         cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
            #                                       ang, vel, mu, rho,q)
            #         del leng, wid, thi, front_index
            #     else:
            #         spatial[q]["GCG"] = mesh.center_mass

            # elif "wing_left" in parameter[q]['CADPART']:
            #     mesh2cad = [[1, 0, 0, -float(parameter[q]["CHORD"]) / 2], [0, 1, 0, -float(parameter[q]["SPAN"]) / 4],
            #                 [0, 0, 1, 0], [0, 0, 0, 1]]
            #     mesh = trimesh.creation.box(extents=(
            #     float(parameter[q]["CHORD"]), float(parameter[q]["SPAN"]) / 2, float(parameter[q]["LASTTWO"])),
            #                                 transform=mesh2cad, center_mass=spatial[q]["CG"])
            #     mesh.apply_transform(Tform)

            #     ecolor = [255 / 255, 0 / 255, 0 / 255]
            #     if include_wing == True:
            #         front_index = np.argmax(
            #             mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
            #         leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
            #         # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
            #         #     np.argmin(mesh.facets_normal[:, 1]), 1]
            #         # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
            #         #     np.argmin(mesh.facets_normal[:, 2]), 2]
            #         # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
            #         #     np.argmin(mesh.facets_normal[:, 0]), 0]
            #         cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
            #                                       ang, vel, mu, rho,q)
            #         del leng, wid, thi, front_index
            #     else:
            #         spatial[q]["GCG"] = mesh.center_mass
            
            #### UAM parts #######
            elif "cyl_ported" in parameter[q]['CADPART']:
                # Flip x and z axis, AKA rotate about y axis
                mesh2cad = [[np.cos(np.pi / 2), 0, np.sin(np.pi / 2), float(parameter[q]["LENGTH"]) / 2], [0, 1, 0, 0],
                            [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2), 0], [0, 0, 0, 1]]

                mesh = trimesh.creation.cylinder(float(parameter[q]["DIAMETER"]) / 2, float(parameter[q]["LENGTH"]),
                                                 sections=10, segment=None, transform=mesh2cad,
                                                 center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)
                cd, cl, cf, warea = cylinderdrag((float(parameter[q]["LENGTH"])) / 1000,
                                                 (float(parameter[q]["DIAMETER"])) / 1000, mesh.symmetry_axis, ang, vel,
                                                 mu, rho)

                ecolor = [62.3 / 255, 174.7 / 255, 238.9 / 255]
                ## structure
                if struct==True:
                    structure[q]["E"] = 71.7e9#Pa ref, https://www.matweb.com/search/DataSheet.aspx?MatGUID=4f19a42be94546b686bbf43f79c51b7d&ckck=1
                    structure[q]["G"] = 26.9e9#Pa
                    structure[q]["type"] = "circular"
                    
                    structure[q]["normal"] = np.round(mesh.symmetry_axis,10)
                    
                    norm = structure[q]["normal"]  * mesh.bounds
                    cent = (mesh.center_mass - mesh.center_mass * structure[q]["normal"])
                    
                    ID = float(structure[q]["DIAMETER"]) - float(structure[q]["WALL_THICKNESS"])
                    OD = float(structure[q]["DIAMETER"])
                  # ### put temporary 6/30/22
                  #   structure[q]["MASS"] = (np.pi*float(structure[q]["LENGTH"]) * (OD**2 - ID**2)/4)* 0.0027 / 1000# densituy in g/mm^3 so ajust
                  
                  # ###
                    
                    structure[q]["p1"] = cent+norm[0]
                    structure[q]["p2"] = cent+norm[1]
                    structure[q]["p3"] = structure[q]["p1"] + np.cross(np.diff(mesh.bounds, axis = 0),structure[q]["normal"])/2#you have the normal figure out how to apply
                    structure[q]["I"] = [[ np.pi/4 * ((OD/2)**4 + (ID/2)**4), 0,0],
                                      [0,np.pi/4 * ((OD/2)**4 + (ID/2)**4), 0],
                                      [0,0,np.pi/2 * ((OD/2)**4 + (ID/2)**4)]] # assume z points along length, per trimesh
                    structure[q]["I"] = trimesh.inertia.transform_inertia(mesh2cad, structure[q]["I"])# Get into the part coordinate system
                    structure[q]["I"] = trimesh.inertia.transform_inertia(Tform, structure[q]["I"])# Get into the assembly coordinate system
                    
                
            elif "aero_shell" in parameter[q]['CADPART']:
                # First figure out, is it plate like or cylinder like
                if float(parameter[q]['CHORD']) < float(parameter[q]['THICKNESS']) / 2:  # then it is like a cylinder
                    # rotate about x axis
                    mesh2cad = [[1, 0, 0, 0],
                                [0, np.cos(np.pi / 2), -np.sin(np.pi / 2), float(parameter[q]["SPAN"]) / 2],
                                [0, np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 0, 1]]

                    mesh = trimesh.creation.cylinder(float(parameter[q]["THICKNESS"]) / 2, float(parameter[q]["SPAN"]),
                                                     sections=10, segment=None, transform=mesh2cad,
                                                     center_mass=spatial[q]["CG"])
                    mesh.apply_transform(Tform)
                    cd, cl, cf, warea = cylinderdrag((float(parameter[q]["SPAN"])) / 1000,
                                                     (float(parameter[q]["THICKNESS"])) / 1000, mesh.symmetry_axis, ang,
                                                     vel, mu, rho)

                else:  # else its a box
                    # mesh2cad = [[1,0,0,float(parameter[q]["THICKNESS"])/2],[0,np.cos(np.pi/2),-np.sin(np.pi/2),float(parameter[q]["SPAN"])/2], [0,np.sin(np.pi/2),np.cos(np.pi/2),-float(parameter[q]["CHORD"])/2], [0,0,0,1]]

                    mesh2cad = [[1, 0, 0, 0], [0, 1, 0, float(parameter[q]["SPAN"]) / 2], [0, 0, 1, 0], [0, 0, 0, 1]]

                    mesh = trimesh.creation.box(extents=(
                    float(parameter[q]["CHORD"]) + float(parameter[q]["THICKNESS"]), float(parameter[q]["SPAN"]),
                    float(parameter[q]["THICKNESS"])), transform=mesh2cad, center_mass=spatial[q]["CG"])
                    mesh.apply_transform(Tform)
                    front_index = np.argmax(
                        mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
                    leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
                    # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
                    #     np.argmin(mesh.facets_normal[:, 1]), 1]
                    # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
                    #     np.argmin(mesh.facets_normal[:, 2]), 2]
                    # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
                    #     np.argmin(mesh.facets_normal[:, 0]), 0]
                    cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
                                                  ang, vel, mu, rho,q)
                ecolor = [122.5 / 255, 12 / 255, 12 / 255]
                if struct==True:
                    structure[q]["E"] = 71.7e9#Pa ref, https://www.matweb.com/search/DataSheet.aspx?MatGUID=4f19a42be94546b686bbf43f79c51b7d&ckck=1
                    structure[q]["G"] = 26.9e9#Pa
                    structure[q]["type"] = "rectangular"

            elif "naca2port" in parameter[q]['CADPART']:
                # rotate about x axis
                mesh2cad = [[1, 0, 0, float(parameter[q]["LENGTH"]) / 2], [0, np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                            [0, np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 0, 1]]
                mesh = trimesh.creation.box(extents=(
                float(parameter[q]["LENGTH"]), float(parameter[q]["PORT_THICKNESS"]),
                float(parameter[q]["THICKNESS"]) / 100 * float(parameter[q]["CHORD"])), transform=mesh2cad,
                                            center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)
                ecolor = [62.3 / 255, 174.7 / 255, 238.9 / 255]

                front_index = np.argmax(mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
                leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
                # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 1]), 1]
                # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 2]), 2]
                # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
                #     np.argmin(mesh.facets_normal[:, 0]), 0]
                
                cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
                                              ang, vel, mu, rho,q)
            ## structure
                if struct==True:
                    structure[q]["E"] = 71.7e9#Pa ref, https://www.matweb.com/search/DataSheet.aspx?MatGUID=4f19a42be94546b686bbf43f79c51b7d&ckck=1
                    structure[q]["G"] = 26.9e9#Pa
                    structure[q]["type"] = "connection" 
                    
            elif "sym_wing_taper" in parameter[q]['CADPART']:
                # Create polygon for potentially tapered wing.
                cs = [float(parameter[q]["CHORD_2"]),float(parameter[q]["CHORD_1"])]
                csi = (cs==np.min(cs))*1
                offset = np.max(cs)* float(parameter[q]["TAPER_OFFSET"]) - np.min(cs) * float(parameter[q]["TAPER_OFFSET"])
                wingperi = trimesh.path.polygons.edges_to_polygons(np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), np.array(
                    [[offset * csi[1], 0], [offset * csi[0], float(parameter[q]["SPAN"])],
                     [float(parameter[q]["CHORD_2"]) + offset * csi[0], float(parameter[q]["SPAN"])],
                     [float(parameter[q]["CHORD_1"])+ offset * csi[1], 0]]))  # For some reason this comes out as a list so choose [0]
                
                wingthic = np.max([float(parameter[q]["CHORD_2"]), float(parameter[q]["CHORD_1"])]) * float(
                    parameter[q]["THICKNESS"]) / 100  # Assume wing to be at max thickness
                # Rotate about x axis
                mesh2cad = [[1, 0, 0, 0], [0, np.cos(np.pi / 2), -np.sin(np.pi / 2), wingthic / 2],
                            [0, np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 0, 1]]

                mesh = trimesh.creation.extrude_polygon(wingperi[0], wingthic, transform=mesh2cad,
                                                        center_mass=spatial[q]["CG"])
                mesh.apply_transform(Tform)
                ecolor = [128.7 / 255, 3.9 / 255, 3.9 / 255]
                
                
                if include_wing == True:
                    front_index = np.argmax(
                        mesh.facets_normal[:, 0])  # Which face is pointing towards flow (along x axis)
                    leng, wid, thi = flow_face(mesh.facets_origin,mesh.facets_normal)
                    # leng = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 1]), 1] - mesh.facets_origin[
                    #     np.argmin(mesh.facets_normal[:, 1]), 1]
                    # wid = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 2]), 2] - mesh.facets_origin[
                    #     np.argmin(mesh.facets_normal[:, 2]), 2]
                    # thi = mesh.facets_origin[np.argmax(mesh.facets_normal[:, 0]), 0] - mesh.facets_origin[
                    #     np.argmin(mesh.facets_normal[:, 0]), 0]
                    cd, cl, cf, warea = platedrag(leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,],
                                                  ang, vel, mu, rho,q)
                    del leng, wid, thi, front_index
                else:
                    spatial[q]["GCG"] = mesh.center_mass
                    
            ## structure
                if struct==True:
                    structure[q]["E"] = 228e9#Pa
                    structure[q]["G"] = 12e9#Pa
                    structure[q]["type"] = "wing"
                    airfoil_area = float(parameter[q]["THICKNESS"])/ 100 * ((float(parameter[q]["CHORD_1"]) + float(parameter[q]["CHORD_2"])) / 2)**2 # span times mean chord
                    structure[q]["normal"] = np.round(mesh.facets_normal[np.argmin(mesh.facets_area - airfoil_area)] ,10) # Points down span
                    
                    norm = structure[q]["normal"]  * mesh.bounds
                    cent = (mesh.center_mass - mesh.center_mass * structure[q]["normal"])
                  ###
                    structure[q]["p1"] = cent+norm[0]
                    structure[q]["p2"] = cent+norm[1]
                    structure[q]["p3"] = structure[q]["p1"] + np.cross(np.diff(mesh.bounds, axis = 0),structure[q]["normal"])/2#you have the normal figure out how to apply
                    
            elif "fuse_sphere_cone" in parameter[q]['CADPART']:
                mesh2cad = [[np.cos(np.pi / 2), 0, np.sin(np.pi / 2), 0], [0, 1, 0, 0],
                            [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2), 0], [0, 0, 0, 1]]
                # First we create a cylinder to get orientation
                mesh = trimesh.creation.cylinder(float(parameter[q]["SPHERE_DIAMETER"]) / 2,
                                                 height=float(parameter[q]["LENGTH"]) - float(
                                                     parameter[q]["SPHERE_DIAMETER"]), transform=mesh2cad,
                                                 center_mass=spatial[q]["CG"], sections=10)

                mesh.apply_transform(Tform)
                cap_norm = mesh.symmetry_axis
                cap_cg = mesh.center_mass
                mesh = trimesh.creation.capsule(
                    height=float(parameter[q]["LENGTH"]) - float(parameter[q]["SPHERE_DIAMETER"]),
                    radius=float(parameter[q]["SPHERE_DIAMETER"]) / 2, count=[10, 10])

                # capsule is strange and won't let me add these in initial step
                mesh.apply_transform(mesh2cad)
                mesh.apply_transform(Tform)

                mesh.center_mass = cap_cg

                ecolor = [162.3 / 255, 174.7 / 255, 238.9 / 255]
                fchord = float(parameter[q]["SPHERE_DIAMETER"]) / 2 + (float(parameter[q]["MIDDLE_LENGTH"])) + (
                            float(parameter[q]["SPHERE_DIAMETER"]) / 2) / ((float(
                    parameter[q]["SPHERE_DIAMETER"]) / 2 - float(parameter[q]["TAIL_DIAMETER"])) / (float(
                    parameter[q]["LENGTH"]) - float(parameter[q]["SPHERE_DIAMETER"]) / 2 - float(
                    parameter[q]["MIDDLE_LENGTH"])))

                cd, cl, cf, warea = ellipticaldrag(fchord / 1000, (float(parameter[q]["LENGTH"])) / 1000,
                                                   (float(parameter[q]["SPHERE_DIAMETER"])) / 1000, cap_norm, ang, vel,
                                                   mu, rho)
            ## structure
                if struct==True:
                    structure[q]["E"] = 71.7e9#Pa ref, https://www.matweb.com/search/DataSheet.aspx?MatGUID=4f19a42be94546b686bbf43f79c51b7d&ckck=1
                    structure[q]["G"] = 26.9e9#Pa
                    structure[q]["type"] = "fuselage"

            elif "naca_fuse_ported" in parameter[q]['CADPART']:
                dianaca = float(parameter[q]["THICKNESS"]) / 100 * float(parameter[q]["CHORD"])
                mesh2cad = [[np.cos(np.pi / 2), 0, np.sin(np.pi / 2), dianaca / 2], [0, 1, 0, 0],
                            [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2), 0], [0, 0, 0, 1]]
                # First we create a cylinder to get orientation

                mesh = trimesh.creation.cylinder(dianaca / 2, height=float(parameter[q]["LENGTH"]) - dianaca,
                                                 transform=mesh2cad, center_mass=spatial[q]["CG"], sections=10)

                mesh.apply_transform(Tform)
                cap_norm = mesh.symmetry_axis
                cap_cg = mesh.center_mass
                mesh = trimesh.creation.capsule(height=float(parameter[q]["LENGTH"]) - dianaca, radius=dianaca / 2,
                                                count=[10, 10])

                # capsule is strange and won't let me add these in initial step
                mesh.apply_transform(mesh2cad)
                mesh.apply_transform(Tform)

                mesh.center_mass = cap_cg

                ecolor = [162.3 / 255, 174.7 / 255, 238.9 / 255]

                cd, cl, cf, warea = ellipticaldrag(float(parameter[q]["CHORD"]) / 1000,
                                                   float(parameter[q]["LENGTH"]) / 1000, dianaca / 1000, cap_norm, ang,
                                                   vel, mu, rho)
            ## structure
                if struct==True:
                    structure[q]["E"] = 71.7e9#Pa ref, https://www.matweb.com/search/DataSheet.aspx?MatGUID=4f19a42be94546b686bbf43f79c51b7d&ckck=1
                    structure[q]["G"] = 26.9e9#Pa
                    structure[q]["type"] = "fuselage"


            else:
                spatial[q]['GCG'] = np.array(spatial[q]['CG'], dtype=float) + np.array(spatial[q]['Translation'],  dtype=float)
                
                
                                                                                       
            if  'mesh' in locals() and struct==True and o==1:# If running structural analysis keep the transformations
                structure[q]["mesh2cad"] = np.array(mesh2cad)
                structure[q]["Tform"] = Tform
                structure[q]["mass"] = float(spatial[q]["MASS"])
                structure[q]["mesh"] = mesh
                structure[q]["GCG"] = mesh.center_mass
                structure[q]["ecolor"] = ecolor
               
                
                                                                            

            if 'mesh' in locals() and create_plot == True:
                ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces, Z=mesh.vertices[:, 2],
                                color=(0, 0, 0, 0), edgecolor=ecolor)
                
            # Add mesh specfic information
            if 'mesh' in locals() and 'cd' in locals():  # "prop" not in parameter[q]['CADPART']and "wing" not in parameter[q]['CADPART']:
                if o == 1 and stl_output == True:
                    Mesh_list.append(mesh)
                # Get projection onto YZ plane as a rectangle
                parea = np.ones([cd.shape[1], 4])
                poly = list()
                for k, qq in enumerate(ang):  # For each AOA
                    N = [np.cos(np.deg2rad(qq)), 0, np.sin(np.deg2rad(qq))]
                    poly.append(trimesh.path.polygons.projected(mesh, N))
                spatial[q]['Cd'] = cd
                spatial[q]["rarea"] = warea * np.ones([1, np.size(ang)])
                spatial[q]["oarea"] = warea
                spatial[q]['Cl'] = cl
                spatial[q]["GCG"] = mesh.center_mass
                spatial[q]['ecolor'] = ecolor
                spatial[q]["mfun"] = np.ones([100,Vel.shape[1]]) #np.ones(np.shape(Vel, 1))
                spatial[q]["marea"] = np.zeros([1, np.size(ang)])
                spatial[q]['Poly'] = poly
                spatial[q]['X_BB'] = np.sort(mesh.bounds[:, 0])[::-1]  # this is the x coordinates for a bounding box, larger x is always towards front / in coming flow
               
                
                if o==1:## Propeller interference loop only do for forward flight o==1
                
                    for qqq in prop: # loop through propellers
                    # Get mesh polygon relative to every propeller normal and store that with the propeller
                        pitem = trimesh.path.polygons.projected(mesh, prop[qqq]["N"])
                        if prop[qqq]["Poly"].intersects(pitem):
                            pinfl = pitem.intersection(prop[qqq]["Poly"])
                            prop[qqq]["Part_Int"].append(q)# keep track of which parts are in the prop slipstream
                            #prop[qqq]["CdS"] = spatial[q]['Cd'] * np.tile(spatial[q]['rarea'],[np.size(Vel, axis=0),1]) * np.tile(modder,[len(vel), 1]) +  prop[qqq]["CdS"]
                            # if intersection is within a prop diameter distance away then it is causing blockage of propeller
                            if distlength([prop[qqq]["GCG"][0],spatial[q]["GCG"][0],prop[qqq]["GCG"][1], spatial[q]["GCG"][1],prop[qqq]["GCG"][2], spatial[q]["GCG"][2]]) < prop [qqq]["Diameter"]*20: #  abs(prop[qqq]["GCG"][0] - spatial[q]["X_BB"][0]) < prop [qqq]["Diameter"] or abs(prop[qqq]["GCG"][0] - spatial[q]["X_BB"][1]) < prop [qqq]["Diameter"]:# consider CG of prop and front or rear face of object
                               prop[qqq]["Blockage_Area"] = prop[qqq]["Blockage_Area"].union(pinfl)#.area + prop[qqq]["Blockage_Area"]
                                ## 8/22/22 we need to use the union!!!!
                            #if prop[qqq]["GCG"][0] >= spatial[q]["GCG"][0]:# propeller is in front of item
                
                            #else:# propeller is rear of item, we make this ratio negative
                                
                            if "wing" in parameter[q]["CADPART"]:# right now (8/3/22) we only care about wings
                                
                                if prop[qqq]["GCG"][0] >= spatial[q]["GCG"][0]:# propeller is in front of item
                                    prop[qqq]["polyp"].append([q,pinfl.area / pitem.area, np.array([spatial[q]['GCG'][0], np.array(pinfl.centroid)[1], np.array(pinfl.centroid)[0]]),-prop[qqq]["N"]])
                                    
                                else:# propeller is rear of item, we make this ratio negative
                                    prop[qqq]["polyp"].append([q,-pinfl.area / pitem.area, np.array([spatial[q]['GCG'][0], np.array(pinfl.centroid)[1], np.array(pinfl.centroid)[0]]),-prop[qqq]["N"]])# need to check this matches the fdm
                                
                                # plt.figure()
                                # trimesh.path.polygons.plot([pitem,prop[qqq]["Poly"], pinfl], label = [q, qqq])
                                # plt.legend()
                                #                         ##### check front and rear of propeller!
                    ######
                            

                if debug == True:
                    spatial[q]["interf"] = list()
                    spatial[q]["interf_mfun"] = list()
                    spatial[q]["interf_IA"] = list()
                    spatial[q]['type'] = np.ones(np.shape(Vel)) - 1
               
                del mesh, cd  # we need to clear this out so we know if something did not have drag

        if o == 1 and stl_output == True:
            Con_Mesh = trimesh.util.concatenate(Mesh_list, b=None)  # concat the mesh list
            Con_Mesh.export('aircraft.stl')
            
            
        ####Do interference
        # Sort from front to back based on forward most face
        interf = copy.deepcopy(spatial)
        # Delete items with no mesh
        for q in list(interf):
            if "Cd" not in interf[q]:
                interf.pop(q)
        interf = dict(sorted(interf.items(), key=lambda item: item[1]["X_BB"][0], reverse=True)) 

        for q in list(interf):  # q is the item doing the influenceing
            for qq, a in enumerate(interf):  # a is the item getting influenced
                for qqq, kk in enumerate(interf[q]['Poly']):  # For each AOA
                    if a == q:  # ignore if we are comparing the same items
                        aft = 0
                    else:
                        aft = spatial[q]["Poly"][qqq].intersects(spatial[a]["Poly"][qqq])

                    if aft == 1:
                        L, AR = charlength(interf[q]['Poly'][
                                               qqq].bounds)  # interference is set by leading object size and aspect ratio
                        if debug == True and qqq == ap:
                            spatial[a]["interf"].append(interf[q]["Name"])

                        xd = distlength(
                            [interf[a]['X_BB'][0], interf[q]['X_BB'][1], interf[a]['GCG'][1], interf[q]['GCG'][1],
                             interf[a]['GCG'][2], interf[q]['GCG'][2]]) / L  # distance between CGs / characteristic length. Is newer rear face to front face method

                        if abs(interf[q]["Poly"][qqq].area) > 0.000001 or abs(interf[a]["Poly"][
                                                                                  qqq].area) > 0.000001:  # this case is to avoid small intersection area, which may arise when an object is completely in the shadow
                            IA = interf[a]["Poly"][qqq].intersection(interf[q]["Poly"][qqq]).area / interf[a]["Poly"][qqq].area  # This is a unitless ratio
                        else:
                            IA = 0

                        if AR > 3:
                            mf = modfun_cyl(xd)
                            if debug == True:
                                spatial[a]['type'][:, qqq] = 2
                        else:
                            mf = modfun_disk(xd)
                            if debug == True:
                                spatial[a]['type'][:, qqq] = 1

                        # if qqq==ap:
                        # print('%s, %s,%s, %s' % (q,a, mf, IA)) # goes with if statement, helps see interference effect

                        if mf == 1:
                            IA = 0

                        if debug == True and qqq == ap:
                            spatial[a]["interf_mfun"].append(1 - (1 - mf) * IA)
                            spatial[a]["interf_IA"].append(IA)

                        if mf < 1 and IA > 0:  # A little odd here because IA=1 says big area overlap, but mf=1 says no influence
                          
                            spatial[a]['marea'][0, qqq] = spatial[a]['marea'][0, qqq] + 1  # Number of times area/region interacts with other objects in flow

                            spatial[a]['mfun'][int(spatial[a]['marea'][0, qqq] - 1), qqq] = 1 - (1 - mf) * IA  # Save modifying scalar, multiply by area of overlap ratio
                            
            interf.pop(q)

        #### Apply interference
        Total_Drag = 0
        Total_Lift = 0
        Total_Cd = 0
        Total_Cl = 0
        Total_Area = 0
        Wing_Drag = 0
        Wing_Lift=0
        Drag_MomentX = np.zeros(np.shape(Vel))
        Drag_MomentY = np.zeros(np.shape(Vel))
        Drag_MomentZ = np.zeros(np.shape(Vel))
        for q in spatial:
            if o==1:
                spatial[q]['Connected'] = list()# Initialize connection info
                if struct==True:
                    structure[q]["Connected"] = list()
            if "Cd" in spatial[q]:

                # Here we have spotlight area  drag  + shadowed area drag contributions
                # spatial[q]["Drag"] = 0.5 * rho * Vel**2 * spatial[q]['Cd'] * np.tile(spatial[q]['rarea'], [np.size(Vel,axis=0),1]) + 0.5 * rho * Vel**2 * spatial[q]['Cd'] * np.tile(spatial[q]['marea'], [np.size(Vel,axis=0),1]) * spatial[q]['mfun']
                modder = np.min(spatial[q]['mfun'], axis=0)

                spatial[q]["Drag"] = 0.5 * rho * Vel ** 2 * spatial[q]['Cd'] * np.tile(spatial[q]['rarea'],[np.size(Vel, axis=0),1]) * np.tile(modder,[len(vel), 1])
                Total_Drag = Total_Drag + spatial[q]["Drag"][vp, ap]
                # I am including the wing drag to find the center of drag, otherwise it be heavily biased if the wing shields a large part of the body.
                Drag_MomentX = Drag_MomentX + spatial[q]["Drag"] * np.ones(np.shape(Vel)) * spatial[q]['GCG'][0]
                Drag_MomentY = Drag_MomentY + spatial[q]["Drag"] * np.ones(np.shape(Vel)) * spatial[q]['GCG'][1]
                Drag_MomentZ = Drag_MomentZ + spatial[q]["Drag"] * np.ones(np.shape(Vel)) * spatial[q]['GCG'][2]

                spatial[q]["Lift"] = 0.5 * rho * Vel ** 2 * spatial[q]['Cl'] * np.tile(spatial[q]['rarea'],
                                                                                       [np.size(Vel, axis=0),
                                                                                        1]) * np.tile(modder,
                                                                                                      [len(vel), 1])
                #0.5 * rho * Vel ** 2 * spatial[q]['Cl'] * np.tile(spatial[q]['rarea'],[np.size(Vel, axis=0),1]) + 0.5 * rho * Vel ** 2 * \spatial[q]['Cl'] * np.tile(spatial[q]['marea'], [np.size(Vel, axis=0), 1]) * \spatial[q]['mfun']
                Total_Lift = Total_Lift + -spatial[q]["Lift"][vp, ap]
                if create_plot == True:
                    ax.arrow3D(spatial[q]["GCG"][0], spatial[q]["GCG"][1], spatial[q]["GCG"][2],
                               -spatial[q]["Drag"][vp, ap] * mp, 0, 0, mutation_scale=20, ec=spatial[q]['ecolor'],
                               fc='red')
                    ax.arrow3D(spatial[q]["GCG"][0], spatial[q]["GCG"][1], spatial[q]["GCG"][2], 0, 0,
                               -spatial[q]["Lift"][vp, ap] * mp, mutation_scale=20, ec=spatial[q]['ecolor'], fc='blue')

                if include_wing == True and "wing" in parameter[q]['CADPART']:
                    Wing_Drag = Wing_Drag + spatial[q]["Drag"][vp, ap]
                    Wing_Lift = Wing_Lift - spatial[q]["Lift"][vp, ap]
                    
                if o==1 and struct==True:# get force outputs for structural model
                    structure[q]["Drag"] = spatial[q]["Drag"]
                    structure[q]["Lift"] = spatial[q]["Lift"]
                    
                if o==1:## Propeller interference loop only do for forward flight o==1
                    for qqq in prop: # loop through propellers
                        for qk in prop[qqq]["Part_Int"]:
                            if qk==q:
                                prop[qqq]["CdS"] = spatial[q]['Cd'] * np.tile(spatial[q]['rarea'],[np.size(Vel, axis=0),1]) * np.tile(modder,[len(vel), 1]) +  prop[qqq]["CdS"]
                                   
                    
                    
        Xfuse = Drag_MomentX[vp, ap] / Total_Drag
        Yfuse = Drag_MomentY[vp, ap] / Total_Drag
        Zfuse = Drag_MomentZ[vp, ap] / Total_Drag
        
        if o == 1:# This is the forward "typical orientation" case
            xfuse = Xfuse.copy()
            yfuse = Yfuse.copy()
            zfuse = Zfuse.copy()
            DragX = Total_Drag - Wing_Drag
            LiftX = Total_Lift - Wing_Lift
            Vx = Vel[vp, ap]
            J_scale=list()
            T_scale= list()
            propo = {}
            #### Propeller interference ####
            for qqq in prop: # loop through propellers
            # Reference for equations: Synthesis of Subsonic Airplane Design by Egbert Torenbeek, page 194
                J_scale.append((qqq, 1 - 0.329 * prop[qqq]["Blockage_Area"].area / prop[qqq]["Diameter"]**2)) # units are in mm and get turned unitless
                T_scale.append((qqq, 1-1.558*(1*prop[qqq]["CdS"][vp,ap] /  prop[qqq]["Diameter"]**2)))
                # if debug ==False:
                # propo[qqq].append(prop[qqq]["polyp"])
                for qqqq in prop[qqq]['polyp']:
                    if qqqq[0] not in propo.keys():
                        propo[qqqq[0]] =[[qqq , qqqq[1]]]
                    else:
                        propo[qqqq[0]].append([qqq , qqqq[1]])
                #propo[qqq] = prop[qqq]['polyp']
                
                # if struct==True:
                #     for q in connect:
                #         if q["FROM_COMP"] in spatial.keys():
                #             spatial[q["FROM_COMP"]]["Connected"].append(q["TO_COMP"])
                #         if q["FROM_COMP"] in structure.keys():
                #             structure[q["FROM_COMP"]]["Connected"].append(q["TO_COMP"])
                
        elif o == 2:
            DragY = Total_Drag - Wing_Drag
            Vy = Vel[vp, ap]
        elif o == 3:
            DragZ = Total_Drag - Wing_Drag
            Vz = Vel[vp, ap]

        if create_plot == True:
            ## add sphere at NP
            NP_mesh = trimesh.creation.uv_sphere(radius=mp / 10, count=[10, 10], theta=None, phi=None)
            NP_mesh.apply_transform([[1, 0, 0, Xfuse], [0, 1, 0, Yfuse], [0, 0, 1, Zfuse], [0, 0, 0, 1]])
            ax.plot_trisurf(NP_mesh.vertices[:, 0], NP_mesh.vertices[:, 1], triangles=NP_mesh.faces,
                            Z=NP_mesh.vertices[:, 2], color=(0, 0, 0, 0), edgecolor=[0, 1, 0])

        if create_plot == True:
            #### Plotting things
            scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.invert_zaxis()
            ax.invert_yaxis()
            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

            ax.set_title('Total Drag = ' + "{:.2f}".format(Total_Drag) + ' N / Total Lift = ' + "{:.2f}".format(
                Total_Lift) + ' N' + ' [NonWingDrag = ' + "{:.2f}".format(Total_Drag - Wing_Drag) + ' N]')

            ax.set_box_aspect((1, 1, 1))
            plt.show()
    

            
    
    xfuseA = (2 * DragX / (rho * Vx ** 2)) * 1000 ** 2
    yfuseA = (2 * DragY / (rho * Vy ** 2)) * 1000 ** 2
    zfuseA = (2 * DragZ / (rho * Vz ** 2)) * 1000 ** 2

    center = [str(xfuse), str(yfuse), str(zfuse)]
    drags = [str(xfuseA), str(yfuseA), str(zfuseA)]
    
   
    if debug==True :
        return (drags, center, spatial, parameter,structure, propo, J_scale, T_scale)
    elif structo==True:
        return(drags, center, structure,propo, J_scale, T_scale)
    else:
        return (drags, center,propo, J_scale, T_scale)

    
if __name__ == "__main__":
    start = time.time()
    DB = False
    struct=False
    if DB==True:
        drags, center, spatial, parameter,structure, prop, J_scale, T_scale = run_full(DataName='designData.json', ParaName='designParameters.json', include_wing=True, create_plot=True,
                 debug=True, stl_output=True, struct=True)
    elif struct==True:
        drags, center, structure, prop, J_scale, T_scale = run_full(DataName='designData.json', ParaName='designParameters.json', include_wing=True, create_plot=False,
                 debug=False, stl_output=False, struct=True)
    else:
        
        drags, center, prop, J_scale, T_scale = run_full(DataName='designData.json', ParaName='designParameters.json', include_wing=True, create_plot=False,
                 debug=False, stl_output=True, struct=False)






    print(drags)
    print(center)

    end = time.time()
    print("Time elapsed:", end - start)
