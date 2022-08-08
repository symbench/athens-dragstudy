# ATHENS Drag Study Tools

This repository contains a cli for generating drag data for UAV and UAM vehicles from the [SwRI Athens Corpus](https://git.isis.vanderbilt.edu/SwRI/athens-client). 

At a high level, this repo runs the [new drag model](https://git.isis.vanderbilt.edu/SwRI/uam_direct2cad/-/blob/main/CoffeeLessDragModel.py) with altered parameters for the specified vehicle. It is currently possible to randomly adjust cylinder/tube length, propeller diameter, and wing shape (chord1, chord2, span) and store the resulting drag values. Note that the drag model requires the existence of the `designData.json` and `designParameters.json` which are generated in [buildcad.py](https://git.isis.vanderbilt.edu/SwRI/uam_direct2cad/-/blob/main/buildcad.py) and are transformations of the 8 json files produced by the [autograph.py](https://git.isis.vanderbilt.edu/SwRI/uam_direct2cad/-/blob/main/AutoGraph.py). *Thus, we assume that the vehicle has been built in Creo at least once.* This is because the drag model needs the transforms (i.e. orientation, rotation, and translation) for each component in the vehicle in order to create representative meshes.

The benefit of this tool is that, instead of updating parameters and rebuilding the vehicle using Creo (slow), we can regenerate the mesh with modified parameter values and record the resulting drag data. Then, for parameters that produce "optimal" drag, we can run the vehicle fewer times through the slower Creo pipeline. Additionally, by operating directly on the dictionaries representing the `designData.json` and `designParameters.json` files, we save a little file I/O by reading from these files once and then writing to the dictionaries on each iteration of the drag model. Currently, to update parameters of interest and run the drag model using this tool takes `~1s / iteration`. 

## Installation & Usage

1. To install this package, run `pip install -e .` from the root of this repository (assumes Python >= 3.8 or so)
2. To store the baseline info for a vehicle including x, y, z plots, the stl mesh, and default drags and centers, run `athens-dragstudy --vehicle <vehicle_name> --baseline`. This will exit if the data already exists for the vehicle's default parameters and assumes that `designData.json` and `designParameters.json` files are present in `data/uav|uam/<vehicle_name>/`.

3. To store results from `n` iterations of the drag model with specified study parameters, run `athens-dragstudy --vehicle <vehicle_name> --rand-length --rand-prop --rand-wing --runs n`
   1. The `rand-length` flag will randomly alter the `LENGTH` of each cylinder/tube in the vehicle, the `rand-prop` flag will alter the `DIAMETER` of each propeller in the vehicle, and the `rand-wing` flag will alter the `CHORD_1`, `CHORD_2`, and `SPAN` values of each wing in the vehicle. 
4. Results are written to `data/uav|uam/<vehicle_name>/results/*.csv` where each row represents a single iteration through the drag model and contains the resulting 6 output values from the drag model iteration and the value of each parameter in that iteration.

5. To add a new vehicle, create a directory named after the vehicle in the appropriate `uam` or `uav` directory and copy the  `designData.json` and `designParameters.json` files (this will be a command line option, eventually). Note `uam` and `uav` are still separated because of CAD part and parameter naming differences

## TODO
- [ ] Add curve fitting
- [ ] Clean up data format for studying multiple params (e.g. length vs length + prop)
- [ ] Add option to add new vehicle from command line
- [x] Specify vehicle, study parameters, number of runs
- [x] Create command option to generate baseline results (original vehicle) with plots and mesh