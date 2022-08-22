import os

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.dirname(__file__), "data")
)
UAM_DATA = os.path.join(DATA_PATH, "uam")
UAV_DATA = os.path.join(DATA_PATH, "uav")

UAM_VEHICLES = os.listdir(UAM_DATA)
UAV_VEHICLES = os.listdir(UAV_DATA)

ALL_VEHICLES = UAV_VEHICLES + UAM_VEHICLES
