#!/usr/bin/env python3

import threading
import sys

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

CARLA_EGG = "/home/ctrl/carla/PythonAPI/carla/dist/carla-0.9.16-py3.8-linux-x86_64.egg"
if CARLA_EGG not in sys.path:
    sys.path.insert(0, CARLA_EGG)

try:
    import carla
except ImportError as exc:
    carla = None
    rospy.logfatal(f"Failed to import CARLA: {exc}")


class BEVVisualizer:
    def __init__(self):
        rospy.init_node("bev_visualizer", anonymous=True)
        if carla is None:
            raise RuntimeError("CARLA Python API unavailable")

        self.scale = rospy.get_param("~map_scale", 0.25)
        self.update_rate = rospy.get_param("~update_rate", 2.0)
        self.max_vehicle_count = rospy.get_param("~max_vehicle_count", 6)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        self.base_image, self.offset, self.extent = self._generate_base_map()
        self.vehicle_states = {}
        self.vehicle_paths = {}
        self.lock = threading.Lock()

        self.image_pub = rospy.Publisher("/carla_multi_vehicle_planner/bev_image", Image, queue_size=1)
        self.vehicle_markers_pub = rospy.Publisher("/carla_multi_vehicle_planner/vehicle_markers", MarkerArray, queue_size=1)
        self.path_markers_pub = rospy.Publisher("/carla_multi_vehicle_planner/path_markers", MarkerArray, queue_size=1)

        for index in range(1, self.max_vehicle_count + 1):
            role = f"ego_vehicle_{index}"
            rospy.Subscriber(f"/carla/{role}/odometry", Odometry, self._odom_cb, callback_args=role)
            rospy.Subscriber(f"/planned_path_{role}", Path, self._path_cb, callback_args=role)

        rospy.Timer(rospy.Duration(1.0 / max(self.update_rate, 0.1)), self._timer_cb)

    def _generate_base_map(self):
        waypoints = self.carla_map.generate_waypoints(2.0)
        if not waypoints:
            raise RuntimeError("Could not generate waypoints for BEV map")

        xs = [wp.transform.location.x for wp in waypoints]
        ys = [wp.transform.location.y for wp in waypoints]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        width = max(1, int((max_x - min_x) / self.scale) + 1)
        height = max(1, int((max_y - min_y) / self.scale) + 1)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        for wp in waypoints:
            px = int((wp.transform.location.x - min_x) / self.scale)
            py = int((wp.transform.location.y - min_y) / self.scale)
            py = height - py - 1
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(image, (px, py), 1, (60, 60, 60), -1)

        return image, (min_x, min_y), (max_x - min_x, max_y - min_y)

    def _world_to_pixel(self, x, y, offset=None):
        origin_x, origin_y = offset if offset else self.offset
        px = int((x - origin_x) / self.scale)
        py = int((y - origin_y) / self.scale)
        py = self.base_image.shape[0] - py - 1
        px = np.clip(px, 0, self.base_image.shape[1] - 1)
        py = np.clip(py, 0, self.base_image.shape[0] - 1)
        return px, py

    def _pixel_to_world(self, px, py):
        x = px * self.scale + self.offset[0]
        y = (self.base_image.shape[0] - py - 1) * self.scale + self.offset[1]
        return x, y

    def _odom_cb(self, msg, role):
        with self.lock:
            self.vehicle_states[role] = msg.pose.pose

    def _path_cb(self, msg, role):
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        with self.lock:
            self.vehicle_paths[role] = points

    def _timer_cb(self, _event):
        with self.lock:
            image = self.base_image.copy()
            markers = MarkerArray()
            path_markers = MarkerArray()
            stamp = rospy.Time.now()
            for index, role in enumerate(sorted(self.vehicle_states.keys())):
                color = self._role_color(role)
                pose = self.vehicle_states[role]
                px, py = self._world_to_pixel(pose.position.x, pose.position.y)
                cv2.circle(image, (px, py), 6, color, -1)

                marker = Marker()
                marker.header = Header(stamp=stamp, frame_id="map")
                marker.ns = "vehicles"
                marker.id = index
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose = pose
                marker.scale.x = marker.scale.y = marker.scale.z = 1.5
                marker.color = ColorRGBA(r=color[2] / 255.0, g=color[1] / 255.0, b=color[0] / 255.0, a=0.9)
                markers.markers.append(marker)

                path = self.vehicle_paths.get(role, [])
                if len(path) >= 2:
                    path_marker = Marker()
                    path_marker.header = Header(stamp=stamp, frame_id="map")
                    path_marker.ns = "paths"
                    path_marker.id = index
                    path_marker.type = Marker.LINE_STRIP
                    path_marker.action = Marker.ADD
                    path_marker.scale.x = 0.5
                    path_marker.color = ColorRGBA(r=color[2] / 255.0, g=color[1] / 255.0, b=color[0] / 255.0, a=0.8)
                    for x, y in path:
                        point = Point(x=x, y=y, z=0.1)
                        path_marker.points.append(point)
                        px2, py2 = self._world_to_pixel(x, y)
                        cv2.circle(image, (px2, py2), 2, color, -1)
                    path_markers.markers.append(path_marker)

            img_msg = Image()
            img_msg.header = Header(stamp=stamp, frame_id="map")
            img_msg.height, img_msg.width = image.shape[:2]
            img_msg.encoding = "bgr8"
            img_msg.step = image.shape[1] * 3
            img_msg.data = image.tobytes()

        self.image_pub.publish(img_msg)
        self.vehicle_markers_pub.publish(markers)
        self.path_markers_pub.publish(path_markers)

    @staticmethod
    def _role_color(role):
        palette = {
            "ego_vehicle_1": (0, 255, 0),
            "ego_vehicle_2": (255, 0, 0),
            "ego_vehicle_3": (0, 0, 255),
            "ego_vehicle_4": (255, 255, 0),
            "ego_vehicle_5": (255, 0, 255),
            "ego_vehicle_6": (0, 255, 255),
        }
        return palette.get(role, (200, 200, 200))


if __name__ == "__main__":
    try:
        BEVVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
