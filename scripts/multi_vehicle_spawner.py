#!/usr/bin/env python3

import math
import random
import sys
import time

import rospy
from geometry_msgs.msg import Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

CARLA_EGG = "/home/ctrl/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg"
if CARLA_EGG not in sys.path:
    sys.path.insert(0, CARLA_EGG)

try:
    import carla
except ImportError as exc:
    rospy.logfatal(f"CARLA import failed: {exc}")
    carla = None


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


class MultiVehicleSpawner:
    def __init__(self):
        rospy.init_node("multi_vehicle_spawner", anonymous=True)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.num_vehicles = rospy.get_param("~num_vehicles", 3)
        self.vehicle_model = rospy.get_param("~vehicle_model", "vehicle.tesla.model3")
        self.enable_autopilot = rospy.get_param("~enable_autopilot", False)
        self.spawn_delay = rospy.get_param("~spawn_delay", 0.5)
        self.target_speed = rospy.get_param("~target_speed", 15.0)
        self.randomize_spawn = rospy.get_param("~randomize_spawn", True)
        self.spawn_seed = rospy.get_param("~spawn_seed", None)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager()

        self.vehicles = []
        self.odom_publishers = {}

        self.spawn_vehicles()
        rospy.on_shutdown(self.cleanup)
        rospy.Timer(rospy.Duration(0.1), self.publish_odometry)

    def spawn_vehicles(self):
        blueprint_library = self.world.get_blueprint_library()
        base_bp = blueprint_library.find(self.vehicle_model)
        if base_bp is None:
            rospy.logwarn(f"Vehicle model {self.vehicle_model} not found; using first available")
            base_bp = blueprint_library.filter("vehicle.*")[0]

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        if self.randomize_spawn:
            rng = random.Random()
            if self.spawn_seed is not None:
                rng.seed(int(self.spawn_seed))
            else:
                rng.seed(time.time())
            rng.shuffle(spawn_points)
            rospy.loginfo(
                "Randomized spawn order with seed %s", self.spawn_seed if self.spawn_seed is not None else "system_time"
            )
        else:
            rospy.loginfo("Randomize spawn disabled; using deterministic order")

        for index in range(self.num_vehicles):
            spawn_point = spawn_points[index % len(spawn_points)]
            role_name = f"ego_vehicle_{index + 1}"
            blueprint = blueprint_library.find(base_bp.id)
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", role_name)

            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            retries = 0
            while vehicle is None and retries < 5:
                retries += 1
                time.sleep(0.2)
                vehicle = self.world.try_spawn_actor(blueprint, spawn_point)

            if vehicle is None:
                rospy.logerr(f"Failed to spawn vehicle {role_name}")
                continue

            vehicle.set_autopilot(self.enable_autopilot, self.traffic_manager.get_port())
            if self.enable_autopilot:
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, max(0, 100 - self.target_speed * 2))

            self.vehicles.append(vehicle)
            topic = f"/carla/{role_name}/odometry"
            self.odom_publishers[role_name] = rospy.Publisher(topic, Odometry, queue_size=10)
            transform = vehicle.get_transform()
            rospy.loginfo(
                "Spawned %s at location (%.1f, %.1f, %.1f) rotation (%.1f, %.1f, %.1f)",
                role_name,
                transform.location.x,
                transform.location.y,
                transform.location.z,
                transform.rotation.roll,
                transform.rotation.pitch,
                transform.rotation.yaw,
            )
            time.sleep(self.spawn_delay)

    def publish_odometry(self, _event):
        stamp = rospy.Time.now()
        for vehicle in self.vehicles:
            if vehicle is None or not vehicle.is_alive:
                continue
            role_name = vehicle.attributes.get("role_name", "")
            publisher = self.odom_publishers.get(role_name)
            if publisher is None:
                continue

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            angular_velocity = vehicle.get_angular_velocity()

            pose = Pose()
            pose.position.x = transform.location.x
            pose.position.y = transform.location.y
            pose.position.z = transform.location.z
            pose.orientation = euler_to_quaternion(
                math.radians(transform.rotation.roll),
                math.radians(transform.rotation.pitch),
                math.radians(transform.rotation.yaw),
            )

            twist = Twist()
            twist.linear = Vector3(velocity.x, velocity.y, velocity.z)
            twist.angular = Vector3(angular_velocity.x, angular_velocity.y, angular_velocity.z)

            odom = Odometry()
            odom.header = Header(stamp=stamp, frame_id="map")
            odom.child_frame_id = f"{role_name}/base_link"
            odom.pose.pose = pose
            odom.twist.twist = twist
            publisher.publish(odom)

    def cleanup(self):
        for vehicle in self.vehicles:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
        self.vehicles.clear()


if __name__ == "__main__":
    try:
        MultiVehicleSpawner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
