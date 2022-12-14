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
import csv
import os
import random
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt

from athens_dragstudy import design_exploration, plots
from athens_dragstudy.utils import DragRunner


def run():
    commands = [
        "explore",
        "coffeeless-drag-model",
        "drag-runner",
        "plots",
        "drag-exploration",
        "fuselage-mass-surrogate"
    ]

    pos = len(sys.argv)
    for cmd in commands:
        if cmd in sys.argv:
            pos = sys.argv.index(cmd) + 1

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "command", help="subcommand to execute", choices=sorted(commands)
    )
    args = parser.parse_args(sys.argv[1:pos])

    if args.command == "explore":
        design_exploration.run(sys.argv[pos:])
    elif args.command == "coffeeless-drag-model":
        from athens_dragstudy import CoffeeLessDragModel

        CoffeeLessDragModel.run(sys.argv[pos:])
    elif args.command == "drag-exploration":
        from athens_dragstudy import drag_exploration

        drag_exploration.run(sys.argv[pos:])
    elif args.command == "plots":
        plots.run(sys.argv[pos:])
    elif args.command == "fuselage-mass-surrogate":
        from athens_dragstudy.surrogates import fuselage_mass_surrogate
        fuselage_mass_surrogate.run(sys.argv[pos:])
    elif args.command == "drag-runner":
        new_parser = argparse.ArgumentParser(
            "DragRunner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        new_parser.add_argument(
            "-subject",
            type=str,
            metavar="vehicle or component name",
            help="the name of the drag subject of interest, can be a component or full design",
        )

        new_parser.add_argument(
            "--baseline",
            action="store_true",
            help="run the drag model for the original vehicle parameters",
        )
        new_parser.add_argument(
            "--length",
            action="store_true",
            help="randomly change connector lengths in the design by a small amount",
        )
        new_parser.add_argument(
            "--wing",
            action="store_true",
            help="randomly change connector lengths in the design by a small amount",
        )

        new_parser.add_argument(
            "--runs", type=int, default=1, help="the number of drag runs to complete"
        )

        new_parser.add_argument(
            "--from-zip", action="store_true", help="run a drag study on a design"
        )
        args = new_parser.parse_args(sys.argv[pos:])

        study_params = ["length"]

        dr = DragRunner(args.vehicle, args.runs, study_params, args.baseline, args.from_zip)
        #dr.set_params_and_run_drag()
        dr.run_dragmodel()


if __name__ == "__main__":
    run()
