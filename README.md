# carla_multi_vehicle_planner

ROS1 package that spawns multiple CARLA vehicles, plans per-vehicle NetworkX A* routes across the HD map, executes Pure Pursuit control, and renders a BEV visualization for RViz.

## Nodes

- `multi_vehicle_spawner.py`: spawns `ego_vehicle_N` actors in CARLA and publishes their odometry.
- `networkx_path_planner.py`: builds a dense waypoint graph, runs A* to random destinations, publishes ROS paths, and draws debug lines.
- `multi_vehicle_controller.py`: applies Pure Pursuit steering/throttle and publishes Ackermann commands.
- `bev_visualizer.py`: paints a BEV image and RViz markers for vehicles/paths with unique colors.

## Usage

```
roslaunch carla_multi_vehicle_planner multi_vehicle_autonomy.launch
```

Start the CARLA simulator (0.9.16) before launching the ROS stack. The nodes expect CARLA to run on `localhost:2000` and the CARLA Python egg to be located at `/home/ctrl/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg`.
# NEVADA
# NEVADA
