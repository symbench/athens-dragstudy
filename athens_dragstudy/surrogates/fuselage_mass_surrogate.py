import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from athens_dragstudy.utils import (get_capsule_massprops_data, list_float,
                                    list_str)


class FuselageMassSurrogate:
    def __init__(
        self,
        data_path: Union[os.PathLike, pathlib.Path] = get_capsule_massprops_data(),
        columns: Optional[List[str]] = None,
    ) -> None:
        assert pathlib.Path(data_path).suffix == ".csv"
        self.data = pd.read_csv(data_path, header=0)
        self.columns = list(
            columns
            or ["HORZ_DIAMETER", "VERT_DIAMETER", "FUSE_CYL_LENGTH", "FLOOR_HEIGHT"]
        )
        assert "mass" in self.data, "No mass column found in dataset"
        self.regressor = None
        self.clf = None

    def fit(self):
        points = (self.data[self.columns]).to_numpy(dtype=float, copy=True)
        targets = self.data["mass"].to_numpy(dtype=float, copy=True)
        seed = 42
        train_points, test_points, train_targets, test_targets = train_test_split(
            points, targets, test_size=0.1, random_state=seed
        )
        self.regressor = PolynomialFeatures(degree=4)
        X_ = self.regressor.fit_transform(train_points)
        self.clf = LinearRegression()
        self.clf.fit(X_, train_targets)
        predict_ = self.regressor.fit_transform(test_points)
        results = self.clf.predict(predict_)

        assert results.shape == test_targets.shape
        mae = np.sum(np.abs(results - test_targets)) / len(test_targets)
        print(
            f"Fuselage Mass interpolation Mean Absolute Error: {mae} with {len(test_targets)} test points"
        )

    def predict(self, params):
        if not self.regressor:
            self.fit()
        params_array = [params[column] for column in self.columns]
        predict_ = self.regressor.fit_transform([params_array])
        target = self.clf.predict(predict_)
        return target


def run(args=None):
    parser = ArgumentParser(
        "Fuselage Mass Surrogate", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-path",
        default=get_capsule_massprops_data(),
        type=str,
        help="The csv file to fit with",
    )
    parser.add_argument(
        "--columns",
        help="Columns to use for fitting",
        default=["HORZ_DIAMETER", "VERT_DIAMETER", "FUSE_CYL_LENGTH", "FLOOR_HEIGHT"],
        type=list_str,
    )
    parser.add_argument(
        "--input",
        default="392.69,200.59,473.675,8.41349999999999",
        help="Sample to predict as a comma separated floats, values are interpreted in the same order as columns",
        type=list_float,
    )

    args = parser.parse_args(args)

    fms = FuselageMassSurrogate(args.data_path, args.columns)
    fms.fit()
    params = {column: value for column, value in zip(args.columns, args.input)}
    mass = fms.predict(params)
    print(f"Predicted mass is {mass} for {params}")
