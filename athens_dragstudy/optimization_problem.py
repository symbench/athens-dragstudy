import json
import os
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem


class DesignOptimization(ElementwiseProblem):
    def __init__(
        self,
        design: "DesignExploration",
        changes: List[Dict],
        params_df: pd.DataFrame,
        save_dir: pathlib.Path,
        nvars: int,
        xl: List,
        xu: List,
    ):
        self.design = design
        self.changes = changes
        super().__init__(n_obj=2, n_constr=3, n_var=nvars, xl=xl, xu=xu)
        self.params_df = params_df
        self.save_dir = save_dir
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        params_set = self.design.propagate_parameters(self.changes, x)
        regeneration_failed = False
        try:
            self.design._regenerate_assembly()
        except Exception as e:
            regeneration_failed = True

        intfs = self.design.interferences()
        intf_volume = 0
        if int(intfs["num_interferences"]):
            intf_volume = sum(
                interference["interference_volume"]
                for interference in intfs.get("interferences", [])
            )

        if regeneration_failed:
            out["F"] = [10000, 10000]
            out["G"] = [10000, 10000, 10000]
        else:
            max_length = 0
            max_width = 0
            for param_name in params_set:
                if param_name.endswith("DISP_LENGTH") and max_length < abs(params_set[param_name]):
                    max_length = params_set[param_name]
                if param_name.endswith("DISP_WIDTH") and max_width < abs(params_set[param_name]):
                    max_width = params_set[param_name]

            gs = [
                params_set["HORZ_DIAMETER"] - max_width * 2,
                (params_set["VERT_DIAMETER"] + params_set["FUSE_CYL_LENGTH"]) - max_length * 2
            ]
            drags, centers = self.design.run_drag_model(save_dir=None)
            fds = self.design._compute_forces_at_reference_velocity(drags)
            fdm_mass_properties = self.design._get_fdm_mass_properties()
            self.params_df.loc[len(self.params_df)] = list(x) + [
                *centers,
                *drags,
                *fds,
                intfs["num_interferences"],
                *fdm_mass_properties,
                None,
            ]
            print(fdm_mass_properties[0], fds[0])
            out["F"] = [fdm_mass_properties[0], fds[0]]
            out["G"] = [intf_volume, *gs]
            self.eval_count += 1
            if self.eval_count % 50 == 0:
                self.params_df.to_csv(self.save_dir / "params.csv")
            # with (snapshot_dir / "intfs.json").open("w") as f:
            #     json.dump(intfs, f, indent=2)

    @staticmethod
    def floor_width(vert_radius, horz_radius, floor_height):
        import numpy as np
        floor_y = vert_radius - floor_height
        floor_x = np.sqrt((1 - (floor_y / vert_radius) ** 2) * (horz_radius**2))
        return 2 * floor_x
