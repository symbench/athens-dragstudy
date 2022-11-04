import json
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def list_str(values):
    splitted = values.split(",")
    assert len(splitted) < 4
    return splitted


def scatter_plot3d(data, labels, color_code_by=None, save_as=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*data, marker="o", s=100, c=color_code_by, cmap="turbo")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    if save_as is None:
        plt.show()
    else:
        plt.savefig(save_as)


def plot_drag_contribution(
    spatial, parameters, total_drag, direction=1, save_name=None
):
    plt.close("all")
    running_sum_drag = 0
    component_drag_dicts = {}
    running_sum_wing_drag = 0
    for component, component_spatial_dict in spatial.items():
        if (
            f"Drag_{direction}" in component_spatial_dict
            and "Cd" in component_spatial_dict
        ):
            component_drag = np.array(component_spatial_dict[f"Drag_{direction}"])[1, 1]
            print(f"{component}_drag_contribution = {component_drag/total_drag * 100}%")
            component_params = parameters[component]
            if "wing" in component_params["CADPART"]:
                running_sum_wing_drag += component_drag
            else:
                component_drag_dicts[component] = component_drag
            running_sum_drag += component_drag
    print(
        total_drag,
        running_sum_drag,
        running_sum_wing_drag,
        running_sum_drag - running_sum_wing_drag,
    )

    assert np.allclose(total_drag, running_sum_drag - running_sum_wing_drag)

    plt.bar(component_drag_dicts.keys(), component_drag_dicts.values())
    plt.legend(ncols=5, loc="upper center")
    plt.gcf().set_size_inches(10, 12)
    if save_name is not None:
        plt.savefig(save_name)
    return component_drag_dicts


def scatter_plot2d(data, labels, color_code_by=None, save_as=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(*data, marker="o", s=100, c=color_code_by, cmap="turbo")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if save_as is None:
        plt.show()
    else:
        plt.savefig(save_as)


def run(args=None):
    parser = ArgumentParser(
        description="plots for sweeped designs",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp-dir", help="The experiment directory")
    parser.add_argument(
        "--plot-choice",
        default="generic-csv",
        choices=[
            "generic-csv",
            "surface-area-vs-drag",
            "components-drag-contribution-avg",
        ],
    )
    parser.add_argument(
        "--columns",
        type=list_str,
        help="Columns from params.csv to plot, comma separated list",
    )
    parser.add_argument(
        "--color-code-by", help="Color code the plots using this column", default=None
    )
    parser.add_argument(
        "--save-as",
        help="Save the generated plot vs. displaying them using matplotlib",
        default=None,
    )
    parser.add_argument("--drag-direction", default="x")

    args = parser.parse_args(args)

    exp_dir = Path(args.exp_dir).resolve()
    params = pd.read_csv(exp_dir / "params.csv")
    if args.plot_choice == "generic-csv":
        columns = args.columns
        columns_data = [params[column] for column in columns]

        if args.color_code_by:
            color_code_by = params[args.color_code_by]
        else:
            color_code_by = None

        if len(args.columns) == 3:
            scatter_plot3d(
                columns_data, columns, color_code_by=color_code_by, save_as=args.save_as
            )
        else:
            scatter_plot2d(
                columns_data, columns, color_code_by=color_code_by, save_as=args.save_as
            )

    elif args.plot_choice == "surface-area-vs-drag":
        sfs = (
            params["capsule_fuselage_VERT_DIAMETER"]
            * params["capsule_fuselage_HORZ_DIAMETER"]
        )
        scatter_plot2d(
            [
                (sfs * np.pi).to_numpy() / 4,
                params[f"fd_{args.drag_direction}"].to_numpy(),
            ],
            labels=[
                f"projected_area_{args.drag_direction}",
                f"fd_{args.drag_direction}",
            ],
            color_code_by=params[args.color_code_by],
            save_as=args.save_as,
        )

    elif args.plot_choice == "components-drag-contribution-avg":
        files_dirs = map(lambda p: Path(p).resolve(), params["files_location"])
        drag_dicts = []
        for j, parent in enumerate(files_dirs):
            drag = params.loc[j][f"fd_{args.drag_direction or 'x'}"]
            directions = {"x": 1, "y": 2, "z": 3}
            parameters_dict = None
            with (parent / "designParameters.json").open("r") as json_file:
                parameters_dict = json.load(json_file)

            with (parent / "spatial.json").open("r") as json_file:

                spatial_dict = json.load(json_file)
                components_drag_dict = plot_drag_contribution(
                    spatial_dict,
                    parameters_dict,
                    total_drag=drag,
                    direction=directions.get(args.drag_direction, "x"),
                    save_name=parent
                    / f"drag_{args.drag_direction or 'x'}_contributions.png",
                )
                components_drag_dict["files"] = str(parent)
                drag_dicts.append(components_drag_dict)
        drags_df = pd.DataFrame.from_records(drag_dicts)
        drags_df.to_csv(args.save_as or exp_dir / "non-wing-drag-contribs.csv")


if __name__ == "__main__":
    run(sys.argv[1:])
