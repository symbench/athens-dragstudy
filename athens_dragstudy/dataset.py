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
from athens_dragstudy import PRIMITIVE_DATA, UAM_DATA, UAV_DATA, MASSPROPS_DATA
import pandas as pd


class Dataset:
    def load_csv():
        pass

    def size():
        pass

    def interp():
        pass


class MassPropsDataset(Dataset):

    def __init__(self, subject: str):
        assert subject in ["uavwing", "capsule"], f"invalid subject {subject}"
        self.subject = subject
        print(f"creating {self.__class__.__name__} for subject {self.subject}")
        self.data_path = os.path.join(MASSPROPS_DATA, subject)
        self.data_file = glob.glob(self.data_path + "/*.csv")[0]  # should only be 1 csv
        self.load_csv()

    def load_csv(self):
        self.massprops_df = pd.read_csv(self.data_file, skip_blank_lines=True).drop_duplicates()
        print("len massprops_df ", len(self.massprops_df))
        #print(self.massprops_df.columns)
        #self.massprops_df = self.massprops_df.iloc[:, 1:]  # remove the subject col
        self.cols = self.massprops_df.columns.drop('subject')
        self.massprops_df[self.cols] = self.massprops_df[self.cols].apply(pd.to_numeric, errors='coerce')
        #print(self.massprops_df.dtypes)
        print(f"{self.size()} data points for subject {self.subject}")

    def size(self):
        return self.massprops_df.shape[0]

    def get_min(self, col_key: str):
        """ return the row index and minimal value of col_key"""
        assert col_key in self.cols, f"invalid column name: {col_key}"
        row_idx = self.massprops_df[col_key].idxmin()
        min_val = self.massprops_df[col_key].iloc[row_idx]  # df.loc[df.index[row_idx], col_key]
        print(f"min value of {col_key} is {min_val} and occurs at {row_idx}")
        return (row_idx, min_val)
    
    def get_max(self, col_key: str):
        """ return the row index and maximal value of col_key"""
        assert col_key in self.cols, f"invalid column name: {col_key}"
        print(len(self.massprops_df[col_key]))
        row_idx = self.massprops_df[col_key].idxmax()
        print(row_idx)
        # max_val = self.massprops_df[col_key].iloc[row_idx]   # df.loc[df.index[row_idx], col_key]
        # print(f"min value of {col_key} is {max_val} and occurs at row {row_idx}")
        # return (row_idx, max_val)


class DragDataset:
    """Aggregate csv files of drag results based on the passed"""

    def __init__(self, drag_subject: str, vehicle_type="uav", is_primitive=False, study_params: list = ['length']):
        # self.drag_subject = drag_subject
        # if is_primitive:
        #     self.datapath = os.path.join(PRIMITIVE_DATA, drag_subject, "results")
        # else:
        #     if vehicle_type == "uav":
        #         self.datapath = os.path.join(UAV_DATA, self.drag_subject, "results")
        #     else:
        #         self.datapath = os.path.join(UAM_DATA, self.drag_subject, "results")
        pass

    def load_csv(self):
        # print(f"load drag csv file")
        # #assert self.drag_subject in ALL_PRIMITIVES + ALL_VEHICLES, f"drag subject {self.drag_subject} not found"
        # import pandas as pd
        
        # data_files = os.listdir(self.datapath)[-1]
        pass

        # df = pd.read_csv('myfile.csv', sep=',', header=None)
        # print(df.values)


if __name__ == "__main__":
    #dd = DragDataset("crossbar")

    wing_data = MassPropsDataset("uavwing")
    #wing_data.get_min('volume')
    wing_data.get_max('volume')