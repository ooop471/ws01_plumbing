# 导入 Python 包
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# 定义节点类
class Talker(Node):
    # 继承 Node 类
    def __init__(self):
        super().__init__('talker__node_py')
        self.count1 = 0
        self.count2 = 0
        self.get_logger().info('talker__node_py 节点启动')

        # 创建发布方
        self.publisher = self.create_publisher(String, 'topic', 10)

        # 创建定时器
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    # 组织消息并发布
    def timer_callback(self):
        msg = String()
        self.count1 += 1

        # 定时器每 0.5 秒触发一次，每 5 次计为一次大命中。
        if self.count1 % 5 == 0:
            self.count2 += 1
            self.get_logger().info('进度: %d' % self.count2)

        # std_msgs/String 只有 data 字段。
        msg.data = 'hit(python)%d big_hit(python)%d' % (
            self.count1,
            self.count2,
        )
        self.publisher.publish(msg)
        self.get_logger().info('发布的消息: "%s"' % msg.data)


def main():
    # 初始化 ROS 2
    rclpy.init()

    # 调用 spin 函数启动循环，传入节点对象
    rclpy.spin(Talker())

    # 释放资源
    rclpy.shutdown()


if __name__ == '__main__':
    main();
