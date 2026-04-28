import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class TurtleBot3SmartTracker(Node):
    def __init__(self):
        super().__init__('turtlebot3_smart_tracker')
        
        # QoS Profile is standard for TurtleBot3 nodes to ensure reliable communication
        qos = QoSProfile(depth=10)
        self.bridge = CvBridge()

        # [Innovation 1: Sensor Fusion] - LiDAR Subscriber
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, qos)
        
        # Camera Subscriber (Following Robotis example structure)
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, qos)
        
        # Velocity Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        # State Variables
        self.is_obstacle_ahead = False
        self.target_found = False
        
        self.get_logger().info("TurtleBot3 Smart Tracker Node has started.")

    def scan_callback(self, msg):
        """
        INNOVATION 1: Obstacle-Aware Tracker
        Logic: Combine LiDAR with Camera. If an object is detected within 0.5m 
        in the front arc, the robot is forced to stop regardless of the ArUco marker.
        """
        # Scan the front 30 degrees (15 left, 15 right)
        # In ROS 2, ranges are represented as an array
        front_ranges = msg.ranges[0:15] + msg.ranges[-15:]
        
        # Filter out 0.0 (error) and inf
        valid_ranges = [r for r in front_ranges if r > 0.1]
        
        if valid_ranges and min(valid_ranges) < 0.5:
            self.is_obstacle_ahead = True
        else:
            self.is_obstacle_ahead = False

    def image_callback(self, msg):
        """
        INNOVATION 2: Search & Lock Mode
        Logic: If the marker is missing, spin 360 to scan. Once 'locked', stop and follow.
        """
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # ArUco Detection Settings (DICT_4X4_50 is standard for TurtleBot3 examples)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(cv_image, aruco_dict, parameters=parameters)
        
        twist = Twist()

        if ids is not None:
            self.target_found = True
            # Calculate the horizontal center of the detected marker
            center_x = np.mean(corners[0][0][:, 0])
            # Calculate error from image center
            error = (cv_image.shape[1] / 2) - center_x

            if self.is_obstacle_ahead:
                self.get_logger().warn("OBSTACLE DETECTED: Safety Stop Engaged.")
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                self.get_logger().info("Target Locked: Following Marker.")
                twist.linear.x = 0.15          # Constant forward speed
                twist.angular.z = error / 500.0 # Proportional steering
        else:
            self.target_found = False
            self.get_logger().info("Target Lost: Initiating 360 Search Scan.")
            # Search Behavior: Spin in place
            twist.linear.x = 0.0
            twist.angular.z = 0.5 

        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBot3SmartTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Standard TurtleBot3 shutdown: Stop the robot before exiting
        stop_twist = Twist()
        node.cmd_vel_pub.publish(stop_twist)
        node.get_logger().info("Shutting down Smart Tracker.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
