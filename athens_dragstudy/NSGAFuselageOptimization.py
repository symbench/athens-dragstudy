from athens_dragstudy.design_exploration import DesignExploration

from pymoo.core.problem import ElementwiseProblem

class DesignOptimization(ElementwiseProblem):
    def __init__(self, design: DesignExploration, changes, n_obj, n_constraints, nvars, xl, xu):
        self.design = design
        self.changes = changes
        super().__init__(
            n_obj=n_obj,
            n_constr=n_constraints,
            n_var=nvars,
            xl=xl,
            xu=xu
        )
        self.save_dir, self.changes = self.design.prepare_experiment(changes)

    def _evaluate(self, x, out, *args, **kwargs):
        self.design.propagate_parameters(changes, x)
        # ToDo: Implement
