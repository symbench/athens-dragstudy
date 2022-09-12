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


import glob
import os
from tkinter import ALL
from athens_dragstudy import ALL_VEHICLES, ALL_PRIMITIVES


class DragDataset:
    """Aggregate csv files of drag results based on the passed"""

    def __init__(self, drag_subject: str, study_params: list = ['length']):
        print(F"create drag dataset")
        self.drag_subject = drag_subject
        print(ALL_PRIMITIVES) 

    def load_csv(self):
        print(f"load drag csv file")
        pass

    def get_min_drag_params():
        pass

    def get_max_drag_params():
        pass


if __name__ == "__main__":
    dd = DragDataset("crossbar")
