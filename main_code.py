import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class SmartTracker(Node):
    def __init__(self):
        super().__init__('smart_tracker')
        
        # 1. Initialize Tools
        self.bridge = CvBridge()
        
        # 2. Subscribers (Listening to the World)
        # We listen to the Camera for ArUco and LiDAR for Obstacles
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # 3. Publisher (Controlling the Robot)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 4. State Variables
        self.obstacle_detected = False
        self.get_logger().info("--- SYSTEM ONLINE: Waiting for Sensors ---")

    def scan_callback(self, msg):
        """ INNOVATION 1: SENSOR FUSION (LiDAR + Camera) """
        # We check the front 40 degrees of the LiDAR scan
        front_ranges = msg.ranges[0:20] + msg.ranges[-20:]
        
        # Filter out 0.0 or infinity readings
        clean_ranges = [r for r in front_ranges if r > 0.1]
        
        if clean_ranges:
            min_dist = min(clean_ranges)
            self.obstacle_detected = min_dist < 0.5  # Safety threshold: 0.5 meters
        else:
            self.obstacle_detected = False

    def image_callback(self, msg):
        """ INNOVATION 2: SEARCH & LOCK MODE """
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Detect ArUco Markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(cv_image, aruco_dict, parameters=parameters)
        
        twist = Twist()

        if ids is not None:
            # --- TARGET LOCKED ---
            center_x = np.mean(corners[0][0][:, 0])
            error_x = (cv_image.shape[1] / 2) - center_x
            
            if self.obstacle_detected:
                self.get_logger().warn("OBSTACLE DETECTED! Safety Stop engaged.")
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                self.get_logger().info("Target Locked: Moving Forward")
                twist.linear.x = 0.15          # Forward speed
                twist.angular.z = error_x / 500.0  # Steer toward center
        else:
            # --- SEARCHING MODE ---
            # If marker is missing, the robot rotates to find it
            self.get_logger().info("Marker Lost: Rotating to Scan...")
            twist.linear.x = 0.0
            twist.angular.z = 0.5  # Spin speed

        self.cmd_pub.publish(twist)

def main():
    rclpy.init()
    node = SmartTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
