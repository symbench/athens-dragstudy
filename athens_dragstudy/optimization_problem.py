import json
import os
import pathlib
from typing import Dict, List

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
        super().__init__(n_obj=2, n_constr=2, n_var=nvars, xl=xl, xu=xu)
        self.params_df = params_df
        self.save_dir = save_dir
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.design.propagate_parameters(self.changes, x)
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
            out["G"] = [10000, 10000]
        else:
            snapshot_dir = self.save_dir / f"{self.eval_count + 1}"
            os.makedirs(snapshot_dir)
            drags, centers = self.design.run_drag_model(snapshot_dir)
            fds = self.design._compute_forces_at_reference_velocity(drags)
            fdm_mass_properties = self.design._get_fdm_mass_properties()
            self.params_df.loc[len(self.params_df)] = list(x) + [
                *centers,
                *drags,
                *fds,
                intfs["num_interferences"],
                *fdm_mass_properties,
                snapshot_dir.resolve(),
            ]

            out["F"] = [fdm_mass_properties[0], fds[0]]
            out["G"] = [intf_volume, 0]
            self.eval_count += 1
            with (snapshot_dir / "intfs.json").open("w") as f:
                json.dump(intfs, f, indent=2)
