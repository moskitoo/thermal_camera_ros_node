#!/usr/bin/env python3

from cv_bridge import CvBridge
# from image_transport_py import ImageTransport
import numpy as np
import rclpy
from rclpy.node import Node
# import irpythermal
from ir_py_thermal import irpythermal
from sensor_msgs.msg import Image


# class MyPublisher(Node):
#     def __init__(self):
#         super().__init__('my_publisher')

#         self.image_transport = ImageTransport(
#             'imagetransport_pub', image_transport='compressed'
#         )
#         self.img_pub = self.image_transport.advertise('camera/image', 10)

#         self.bridge = CvBridge()


#         self.camera: irpythermal.Camera

#         camera_kwargs = {}
#         camera_kwargs['camera_raw'] = True
#         self.camera = irpythermal.Camera(**camera_kwargs)

#         timer_period = 0.5
#         self.timer = self.create_timer(timer_period, self.timer_callback)

#     def timer_callback(self):
#         # original = np.uint8(np.random.randint(0, 255, size=(640, 480, 3)))

#         frame = self.camera.get_frame()
#         frame = frame.astype(np.uint8)

#         image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
#         image_msg.header.stamp = self.get_clock().now().to_msg()
#         image_msg.header.frame_id = 'camera'

#         self.img_pub.publish(image_msg)
#         self.get_logger().info('Publishing image')

class MyPublisher(Node):
    def __init__(self):
        super().__init__('my_publisher')
        
        self.img_pub = self.create_publisher(Image, 'camera/image', 10)

        self.bridge = CvBridge()

        camera_kwargs = {'camera_raw': True}
        self.camera = irpythermal.Camera(**camera_kwargs)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        frame = self.camera.get_frame().astype(np.uint8)

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = 'camera'

        self.img_pub.publish(image_msg)
        self.get_logger().info('Publishing image')


def main(args=None):
    rclpy.init(args=args)
    node = MyPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly (optional)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
