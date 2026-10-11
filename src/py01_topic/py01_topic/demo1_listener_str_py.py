# 导入 Python 包
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 定义节点类
class Listener(Node):
    # 继承 Node 类
    def __init__(self):
        super().__init__('listener__node_py')
        self.get_logger().info('listener__node_py 节点启动')

        # 创建订阅方
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10,
        )

    def listener_callback(self, msg):
        self.get_logger().info('收到的消息: "%s"' % msg.data)


def main():
    # 初始化 ROS 2
    rclpy.init()

    # 调用 spin 函数启动循环，传入节点对象
    rclpy.spin(Listener())

    # 释放资源
    rclpy.shutdown()


if __name__ == '__main__':
    main()
