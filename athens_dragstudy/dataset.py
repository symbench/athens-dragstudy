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
import numpy as np


class DragDataset:
    def __init__(vehicle: str, study_params: list = ["length"]):
        pass

    def collect_data(self):
        pass
        #  glob all files of relevant parameters and form into large numpy array
        #  separate input parameters and drag values as x, y
        #  return x, y