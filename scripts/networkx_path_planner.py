#!/usr/bin/env python3

import math
import random
import threading
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header

from setup_carla_path import CARLA_EGG, AGENTS_ROOT

try:
    import carla
except ImportError as exc:
    rospy.logfatal(f"Failed to import CARLA package: {exc}")
    carla = None
    GlobalRoutePlanner = None
    GlobalRoutePlannerDAO = None
else:
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ImportError as exc:
        rospy.logfatal(f"Failed to import CARLA GlobalRoutePlanner: {exc}")
        GlobalRoutePlanner = None
    try:
        from agents.navigation.global_route_planner_dao import (
            GlobalRoutePlannerDAO,
        )
    except ImportError:
        GlobalRoutePlannerDAO = None


class NetworkXPathPlanner:
    """Route planner that leverages CARLA's GlobalRoutePlanner to stay on-lane."""

    def __init__(self):
        rospy.init_node("networkx_path_planner", anonymous=True)
        if carla is None or GlobalRoutePlanner is None:
            raise RuntimeError("CARLA navigation modules unavailable")

        # Parameters
        self.num_vehicles = rospy.get_param("~num_vehicles", 3)
        self.path_sampling = rospy.get_param("~path_sampling", 1.0)
        self.path_update_interval = rospy.get_param("~path_update_interval", 2.0)
        self.min_destination_distance = rospy.get_param("~min_destination_distance", 80.0)
        self.max_destination_distance = rospy.get_param("~max_destination_distance", 400.0)
        self.destination_retry_limit = rospy.get_param("~destination_retry_limit", 20)
        self.destination_reached_threshold = rospy.get_param("~destination_reached_threshold", 5.0)
        self.visualization_lifetime = rospy.get_param("~visualization_lifetime", 60.0)
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        self.global_route_resolution = rospy.get_param("~global_route_resolution", 2.0)
        self.override_preempt_min_dist = float(
            rospy.get_param("~override_preempt_min_dist", 0.0)
        )
        if self.override_preempt_min_dist < 0.0:
            rospy.logwarn("override_preempt_min_dist < 0, clamping to 0.0")
            self.override_preempt_min_dist = 0.0

        # Connect to CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        # Prepare global route planner
        if GlobalRoutePlannerDAO is not None:
            dao = GlobalRoutePlannerDAO(self.carla_map, self.global_route_resolution)
            self.route_planner = GlobalRoutePlanner(dao)
        else:
            self.route_planner = GlobalRoutePlanner(
                self.carla_map, self.global_route_resolution
            )
        if hasattr(self.route_planner, "setup"):
            self.route_planner.setup()

        # Destination candidates
        self.spawn_points: List = self.carla_map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        # Vehicle state
        self.vehicles: List = []
        self.vehicle_paths: Dict[str, List[Tuple[float, float]]] = {}
        self.vehicle_destinations: Dict[str, int] = {}
        self.previous_paths: Dict[str, List[Tuple[float, float]]] = {}
        self.active_destinations: set[int] = set()
        self.override_goal: Dict[str, Optional[Tuple[float, float, rospy.Time]]] = {
            self._role_name(i): None for i in range(self.num_vehicles)
        }
        self._planning_lock = threading.RLock()

        # Publishers
        self.path_publishers: Dict[str, rospy.Publisher] = {}
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/planned_path_{role}"
            self.path_publishers[role] = rospy.Publisher(topic, Path, queue_size=1, latch=True)

        self.override_subscribers = []
        for index in range(self.num_vehicles):
            role = self._role_name(index)
            topic = f"/override_goal/{role}"
            sub = rospy.Subscriber(topic, PoseStamped, self._make_override_cb(role), queue_size=1)
            self.override_subscribers.append(sub)

        self.colors = [
            carla.Color(r=255, g=0, b=0),
            carla.Color(r=0, g=255, b=0),
            carla.Color(r=0, g=0, b=255),
            carla.Color(r=255, g=255, b=0),
            carla.Color(r=255, g=0, b=255),
            carla.Color(r=0, g=255, b=255),
        ]

        # Initial planning
        rospy.sleep(1.0)
        self._refresh_vehicles()
        self._plan_for_all()

        rospy.Timer(rospy.Duration(self.path_update_interval), self._timer_cb)
        rospy.on_shutdown(self._cleanup)

    # ------------------------------------------------------------------
    # Core planning loop
    # ------------------------------------------------------------------
    def _timer_cb(self, _event):
        with self._planning_lock:
            self._refresh_vehicles()
            self._plan_for_all()

    def _refresh_vehicles(self):
        actors = self.world.get_actors().filter("vehicle.*")
        vehicles = []
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role.startswith("ego_vehicle_"):
                vehicles.append(actor)
        vehicles.sort(key=lambda veh: veh.attributes.get("role_name", ""))
        self.vehicles = vehicles
        rospy.loginfo_throttle(5.0, f"Tracking {len(self.vehicles)} ego vehicles")

    def _plan_for_all(self):
        for index, vehicle in enumerate(self.vehicles[: self.num_vehicles]):
            role = self._role_name(index)
            override_goal = self.override_goal.get(role)
            if override_goal is not None:
                if self._check_override_goal(vehicle, role, override_goal):
                    continue
                self._plan_override(vehicle, index, override_goal)
                continue

            reached = self._check_destination(vehicle, role)
            if reached or role not in self.vehicle_paths:
                self._plan_for_vehicle(vehicle, index)

    # ------------------------------------------------------------------
    # Per vehicle planning using GlobalRoutePlanner
    # ------------------------------------------------------------------
    def _plan_for_vehicle(self, vehicle, index: int):
        role = self._role_name(index)

        self._reset_destination_tracking(role)

        start_waypoint = self.carla_map.get_waypoint(
            vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if start_waypoint is None:
            rospy.logwarn(f"{role}: could not resolve start waypoint")
            return

        dest_index, route_points = self._sample_destination_route(start_waypoint)
        if dest_index is None or not route_points:
            rospy.logwarn(f"{role}: failed to find valid destination route")
            return

        self.vehicle_paths[role] = route_points
        self.vehicle_destinations[role] = dest_index
        self.active_destinations.add(dest_index)

        rospy.loginfo(
            "%s: planned CARLA route with %d samples (%.1fm)",
            role,
            len(route_points),
            self._path_length(route_points),
        )

        self._draw_path(route_points, index)
        self._publish_path(route_points, role)

    def _sample_destination_route(
        self, start_waypoint
    ) -> Tuple[Optional[int], Optional[List[Tuple[float, float]]]]:
        start_loc = start_waypoint.transform.location
        attempts = 0
        while attempts < self.destination_retry_limit:
            attempts += 1
            dest_index = random.randrange(len(self.spawn_points))
            if dest_index in self.active_destinations:
                continue

            dest_loc = self.spawn_points[dest_index].location
            euclid = start_loc.distance(dest_loc)
            if euclid < self.min_destination_distance or euclid > self.max_destination_distance:
                continue

            route = self.route_planner.trace_route(start_loc, dest_loc)
            if not route or len(route) < 2:
                continue

            route_length = self._route_length(route)
            if route_length < self.min_destination_distance:
                continue

            # Allow the planner to exceed max_distance slightly since road length > Euclid
            if route_length > self.max_destination_distance * 1.5:
                continue

            sampled_points = self._route_to_points(route)
            if len(sampled_points) < 2:
                continue

            return dest_index, sampled_points

        return None, None

    def _plan_override(
        self,
        vehicle,
        index: int,
        goal_info: Tuple[float, float, rospy.Time],
    ):
        role = self._role_name(index)
        goal_xy = (goal_info[0], goal_info[1])
        start_waypoint = self.carla_map.get_waypoint(
            vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if start_waypoint is None:
            rospy.logwarn(f"{role}: override planning failed to resolve start waypoint")
            return

        goal_waypoint = self._snap_to_waypoint(goal_xy)
        if goal_waypoint is None:
            rospy.logwarn(
                "%s: override goal (%.1f, %.1f) is off road; ignoring",
                role,
                goal_xy[0],
                goal_xy[1],
            )
            return

        start_loc = start_waypoint.transform.location
        goal_loc = goal_waypoint.transform.location
        route = self.route_planner.trace_route(start_loc, goal_loc)
        if not route or len(route) < 2:
            rospy.logwarn(f"{role}: failed to trace override route")
            return

        sampled_points = self._route_to_points(route)
        if len(sampled_points) < 2:
            rospy.logwarn(f"{role}: override route produced insufficient samples")
            return

        self.vehicle_paths[role] = sampled_points

        rospy.loginfo(
            "%s: planned override route with %d samples (%.1fm)",
            role,
            len(sampled_points),
            self._path_length(sampled_points),
        )

        self._draw_path(sampled_points, index)
        self._publish_path(sampled_points, role)
        rospy.loginfo(
            "override path published for %s (%d poses)", role, len(sampled_points)
        )

    # ------------------------------------------------------------------
    # Destination monitoring
    # ------------------------------------------------------------------
    def _check_destination(self, vehicle, role: str) -> bool:
        dest_index = self.vehicle_destinations.get(role)
        if dest_index is None or dest_index >= len(self.spawn_points):
            return False

        dest_location = self.spawn_points[dest_index].location
        current_location = vehicle.get_location()
        distance = current_location.distance(dest_location)
        if distance <= self.destination_reached_threshold:
            rospy.loginfo(f"{role}: destination reached ({distance:.2f} m)")
            self._clear_path(role)
            self.vehicle_paths.pop(role, None)
            self.vehicle_destinations.pop(role, None)
            self.active_destinations.discard(dest_index)
            return True
        return False

    def _check_override_goal(
        self,
        vehicle,
        role: str,
        goal_info: Tuple[float, float, rospy.Time],
    ) -> bool:
        goal_xy = (goal_info[0], goal_info[1])
        current_location = vehicle.get_location()
        distance = math.hypot(
            current_location.x - goal_xy[0], current_location.y - goal_xy[1]
        )
        threshold = max(
            self.destination_reached_threshold, self.override_preempt_min_dist
        )
        if distance <= threshold:
            rospy.loginfo("override goal reached for %s, clearing override", role)
            self._clear_path(role)
            self.vehicle_paths.pop(role, None)
            self.override_goal[role] = None
            return True
        return False

    # ------------------------------------------------------------------
    # ROS / Visualization helpers
    # ------------------------------------------------------------------
    def _publish_path(self, path_points: List[Tuple[float, float]], role: str):
        publisher = self.path_publishers.get(role)
        if publisher is None:
            return
        header = Header(stamp=rospy.Time.now(), frame_id="map")
        msg = Path(header=header)
        for x, y in path_points:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        publisher.publish(msg)

    def _draw_path(self, path_points: List[Tuple[float, float]], index: int):
        if not self.enable_visualization or len(path_points) < 2:
            return
        role = self._role_name(index)
        self._clear_path(role)
        color = self.colors[index % len(self.colors)]
        for i in range(len(path_points) - 1):
            p1 = carla.Location(x=path_points[i][0], y=path_points[i][1], z=0.5)
            p2 = carla.Location(x=path_points[i + 1][0], y=path_points[i + 1][1], z=0.5)
            self.world.debug.draw_line(
                p1,
                p2,
                thickness=0.3,
                color=color,
                life_time=self.visualization_lifetime,
            )
        self.previous_paths[role] = path_points[:]

    def _clear_path(self, role: str):
        if role not in self.previous_paths:
            return
        old_path = self.previous_paths.pop(role)
        clear_color = carla.Color(r=0, g=0, b=0)
        for i in range(len(old_path) - 1):
            p1 = carla.Location(x=old_path[i][0], y=old_path[i][1], z=0.5)
            p2 = carla.Location(x=old_path[i + 1][0], y=old_path[i + 1][1], z=0.5)
            self.world.debug.draw_line(
                p1,
                p2,
                thickness=0.3,
                color=clear_color,
                life_time=0.1,
            )

    def _cleanup(self):
        for role in list(self.previous_paths.keys()):
            self._clear_path(role)
        self.previous_paths.clear()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _make_override_cb(self, role: str):
        def _callback(msg: PoseStamped):
            self._handle_override(role, msg)

        return _callback

    def _handle_override(self, role: str, msg: PoseStamped):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        goal_info = (msg.pose.position.x, msg.pose.position.y, stamp)
        with self._planning_lock:
            if role not in self.override_goal:
                self.override_goal[role] = None
            self.override_goal[role] = goal_info
            self._reset_destination_tracking(role)
            self.vehicle_paths.pop(role, None)
            self._clear_path(role)
            rospy.loginfo(
                "override set for %s: (%.1f, %.1f)", role, goal_info[0], goal_info[1]
            )

            self._refresh_vehicles()
            for index, vehicle in enumerate(self.vehicles[: self.num_vehicles]):
                if self._role_name(index) == role:
                    self._plan_override(vehicle, index, goal_info)
                    break

    def _reset_destination_tracking(self, role: str):
        previous_dest = self.vehicle_destinations.pop(role, None)
        if previous_dest is not None:
            self.active_destinations.discard(previous_dest)

    def _snap_to_waypoint(self, point_xy: Tuple[float, float]):
        location = carla.Location(x=point_xy[0], y=point_xy[1], z=0.5)
        return self.carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    def _route_to_points(
        self, route: List[Tuple]
    ) -> List[Tuple[float, float]]:
        waypoints = [item[0] for item in route]
        if not waypoints:
            return []
        points: List[Tuple[float, float]] = []
        last_point: Optional[Tuple[float, float]] = None
        for wp in waypoints:
            loc = wp.transform.location
            current = (loc.x, loc.y)
            if last_point is None:
                points.append(current)
                last_point = current
                continue
            segment_length = self._distance(current, last_point)
            if segment_length < 1e-3:
                continue
            steps = max(1, int(segment_length // self.path_sampling))
            for step in range(1, steps + 1):
                ratio = min(1.0, (step * self.path_sampling) / segment_length)
                interp = (
                    last_point[0] + (current[0] - last_point[0]) * ratio,
                    last_point[1] + (current[1] - last_point[1]) * ratio,
                )
                if not points or self._distance(interp, points[-1]) > 0.05:
                    points.append(interp)
            if self._distance(current, points[-1]) > 0.05:
                points.append(current)
            last_point = current
        return points

    @staticmethod
    def _route_length(route: List[Tuple]) -> float:
        total = 0.0
        previous = None
        for waypoint, _ in route:
            loc = waypoint.transform.location
            if previous is not None:
                total += loc.distance(previous)
            previous = loc
        return total

    @staticmethod
    def _distance(pt_a: Tuple[float, float], pt_b: Tuple[float, float]) -> float:
        return math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])

    @staticmethod
    def _path_length(points: List[Tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(points) - 1):
            total += math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        return total

    @staticmethod
    def _role_name(index: int) -> str:
        return f"ego_vehicle_{index + 1}"


if __name__ == "__main__":
    try:
        planner = NetworkXPathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
