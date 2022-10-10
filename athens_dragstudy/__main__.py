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
from athens_dragstudy.utils import DragRunner

import time
import csv
import os
import random
import time



# # TODO: if you want a new vehicle to study it must go here
# UAM_VEHICLES = [
#     "spade",
#     "tbar_full",
#     "tbar_double",
#     "tbar_single",
#     "tbar_minime",
#     "tbar_steerwing",
#     "dummy",
# ]

# print(UAM_VEHICLES)

# UAV_VEHICLES = ["axe", "quad"]
# VALID_VEHICLES = UAM_VEHICLES + UAV_VEHICLES



def run():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument('--corpus', type=str, metavar='corpus type',
    #                     choices=['uav', 'uam'])
    parser.add_argument(
        "-subject",
        type=str,
        metavar="vehicle or component name",
        help="the name of the drag subject of interest, can be a component or full design",
    )
    # parser.add_argument(
    #     "--fit", action="store_true", help="fit the specified vehicle to"
    # )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="run the drag model for the original vehicle parameters",
    )
    parser.add_argument(
        "--length",
        action="store_true",
        help="randomly change connector lengths in the design by a small amount",
    )
    parser.add_argument(
        "--wing",
        action="store_true",
        help="randomly change connector lengths in the design by a small amount",
    )
    # parser.add_argument(
    #     "--prop",
    #     action="store_true",
    #     help="randomly change connector lengths in the design by a small amount",
    # )
    parser.add_argument(
        "--runs", type=int, default=1, help="the number of drag runs to complete"
    )

    parser.add_argument(
        "--from-zip", action="store_true", help="run a drag study on a design"
    )

    # subparsers = parser.add_subparsers()
    # parser_prim = subparsers.add_parser("prim")  # specify individual structures
    # parser_prim.add_argument("-struct", type=str) #choices=['rail', 'crossbar', 'tbar', 'fuselage'])

    #parser_fit = subparsers.add_parser("fit")  # specify a fit method for drag data

    args = parser.parse_args()
    #prim_args = parser_prim.parse_args()

    print(f"args are: {args}")

    # if args.fit and args.vehicle:
    #     fit.run(args.vehicle)
    #     sys.exit(0)
    # elif args.fit and not args.vehicle:
    #     print("need to specify a vehicle to fit its data")
    #     sys.exit(0)

    study_params = ["length"]
    # if args.length:
    #     study_params.append("length")
    # # if args.prop:
    # #     study_params.append("prop")
    # if args.wing:
    #     study_params.append("wing")
    print(f"study params are: {study_params}")

    # if args.struct:
    #     dr = DragRunner(args.vehicle, args.runs, study_params, args.baseline, args.from_zip, args.struct)
    # else:
    dr = DragRunner(args.subject, args.runs, study_params, args.baseline, args.from_zip)

    #dr.set_params_and_run_drag()
    dr.run_dragmodel()


if __name__ == "__main__":
    run()
