#!/usr/bin/env python3
"""双目深度发布节点: 读入左右图 -> 算视差和深度 -> 发布。

两种输入模式(靠参数自动判断, 二选一):
  * 图片模式: 给出 left_image / right_image 两个图片路径, 节点按 period
    周期循环计算并发布, 不需要相机。
  * 话题模式: 不给图片路径, 就订阅左右图像话题, 接真实双目相机。
"""
import cv2
import message_filters
import numpy as np
import rclpy
# 导入话题接口
from base_interfaces_demo.msg import Hit
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header


# 定义节点类
class StereoDepthPublisher(Node):
    """Compute metric depth from approximately synchronized stereo images."""
    # 依旧提示加路径，数据填写
    def __init__(self) -> None:
        super().__init__("stereo_depth_publisher")
        # 你也创建结构体
        self.declare_parameter("left_image", "")
        self.declare_parameter("right_image", "")
        self.declare_parameter("left_topic", "/stereo/left/image_raw")
        self.declare_parameter("right_topic", "/stereo/right/image_raw")
        self.declare_parameter("depth_topic", "/stereo/depth")
        self.declare_parameter("info_topic", "/stereo/depth_info")
        self.declare_parameter("frame_id", "stereo_camera")
        self.declare_parameter("period", 1.0)
        self.declare_parameter("fx", 1000.0)
        self.declare_parameter("baseline", 0.1)
        self.declare_parameter("num_disparities", 192)
        self.declare_parameter("block_size", 5)
        # 读取结构体
        left_image = self.get_parameter("left_image").value
        right_image = self.get_parameter("right_image").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("info_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        period = float(self.get_parameter("period").value)
        self.fx = float(self.get_parameter("fx").value)
        self.baseline = float(self.get_parameter("baseline").value)
        num_disparities = max(
            16, int(self.get_parameter("num_disparities").value) // 16 * 16
        )
        block_size = int(self.get_parameter("block_size").value)
        block_size = max(3, block_size if block_size % 2 else block_size + 1)

        if self.fx <= 0.0 or self.baseline <= 0.0:
            raise ValueError("fx and baseline must be positive")
        if period <= 0.0:
            raise ValueError("period must be positive")
        if bool(left_image) != bool(right_image):
            raise ValueError("left_image 和 right_image 必须同时设置")
        # sgbm数据设置
        p1 = 8 * block_size * block_size
        p2 = 32 * block_size * block_size
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.bridge = CvBridge()
        # 创建发布方
        self.depth_publisher = self.create_publisher(Image, depth_topic, 10)
        self.info_publisher = self.create_publisher(Hit, info_topic, 10)
        # 图片模式: 读一次图, 按 period 循环计算; 否则走话题模式订阅左右图
        if left_image and right_image:
            self.left_gray = self.load_gray(left_image)
            self.right_gray = self.load_gray(right_image)
            if self.left_gray.shape != self.right_gray.shape:
                raise ValueError("左右图尺寸不一致, 请使用同一相机拍摄的校正图")
            self.create_timer(period, self.file_timer_callback)
            self.get_logger().info(
                f"图片模式: {left_image} + {right_image} -> {depth_topic}, "
                f"周期 {period}s"
            )
        else:
            left_topic = self.get_parameter("left_topic").value
            right_topic = self.get_parameter("right_topic").value
            # 创建订阅方
            left_sub = message_filters.Subscriber(self, Image, left_topic)
            right_sub = message_filters.Subscriber(self, Image, right_topic)
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [left_sub, right_sub], queue_size=10, slop=0.05
            )
            self.sync.registerCallback(self.image_callback)
            self.get_logger().info(
                f"话题模式: {left_topic} + {right_topic} -> {depth_topic}"
            )

    def load_gray(self, path: str) -> np.ndarray:
        """读取图片并转成灰度图(SGBM 需要灰度输入)。"""
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"读图失败: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 左右同步回调函数(话题模式)
    def image_callback(self, left_msg: Image, right_msg: Image) -> None:
        """Convert a synchronized pair, calculate depth, and publish both messages."""
        # 转换左右图像为OpenCV格式
        try:
            left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="mono8")
            right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="mono8")
        except (ValueError, cv2.error) as exc:
            self.get_logger().error(f"无法转换左右图像: {exc}")
            return

        if left.shape != right.shape:
            self.get_logger().error(
                f"左右图尺寸不一致: {left.shape} vs {right.shape}"
            )
            return
        self.publish_depth(left, right, left_msg.header)

    # 图片模式: 每 period 秒把同一对图片再算一遍
    def file_timer_callback(self) -> None:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        self.publish_depth(self.left_gray, self.right_gray, header)

    # 计算深度
    def publish_depth(self, left, right, header) -> None:
        """视差 -> 深度 -> 发布深度图和统计信息。"""
        disparity = self.matcher.compute(left, right).astype(np.float32) / 16.0
        valid = np.isfinite(disparity) & (disparity > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(
                valid,
                self.fx * self.baseline / np.maximum(disparity, 1e-6),
                np.nan,
            ).astype(np.float32)

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header = header
        self.depth_publisher.publish(depth_msg)

        # 发布深度信息
        valid_depth = depth[valid]
        info = Hit()
        info.name = "stereo_depth"
        info.number = int(valid_depth.size)
        info.depth = float(np.median(valid_depth)) if valid_depth.size else 0.0
        self.info_publisher.publish(info)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StereoDepthPublisher()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
