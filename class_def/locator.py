import serial.tools.list_ports  # 必须导入serial库才能使用list_ports
from geometry_msgs.msg import PoseStamped, TwistStamped
from utils import State, rtls_data_recv, UWBData, read_modbus_frame
import numpy as np
from tf import transformations
from pymodbus.client import ModbusSerialClient
import threading
import time
import rospy


class MotionCapture:
    def __init__(self, name):
        # 暂存的节点状态
        self.state = State()
        self.name = name
        self.subscriber_pose = None
        self.subscriber_twist = None

    # 启动定位
    def run(self):
        # 启动ros
        pose_topic = "/vrpn_client_node/{}/pose".format(self.name)  # 订阅vrpn_PoseStamped话题
        self.subscriber_pose = rospy.Subscriber(pose_topic, PoseStamped, self.update_pos_ang)
        twist_topic = "/vrpn_client_node/{}/twist".format(self.name)  # 订阅vrpn_TwistStamped话题
        self.subscriber_twist = rospy.Subscriber(twist_topic, TwistStamped, self.update_vel)

    # 关闭定位
    def stop(self):
        self.subscriber_pose.unregister()
        self.subscriber_twist.unregister()

    # 读出节点状态
    def __call__(self, *args, **kwargs):
        return self.state

    def update_pos_ang(self, pose_stamped):
        """
        更新位置、角度
        :param pose_stamped: 位置、角度数据
        :return:
        """
        # 位置和角度，动捕中顺序为[z, x, y]
        pos = (pose_stamped.pose.position.z, pose_stamped.pose.position.x, pose_stamped.pose.position.y)
        ang = transformations.euler_from_quaternion([
            pose_stamped.pose.orientation.z,
            pose_stamped.pose.orientation.x,
            pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.w,
        ])
        self.state.pos = np.array(pos)
        self.state.ang = np.array(ang)

    # 更新速度、加速度
    def update_vel(self, twist_stamped):
        """
        更新速度
        :param twist_stamped: 速度数据
        :return:
        """
        # 速度，动捕中顺序为[z, x, y]
        vel = (twist_stamped.twist.linear.z, twist_stamped.twist.linear.x, twist_stamped.twist.linear.y)
        self.state.vel = np.array(vel)


class UWBLocator:
    def __init__(self, tag_id, slave, mode='ANCHOR'):
        self.state = State()
        self.client = None
        self.port = None  # 最终选定的串口
        self.tag_id = tag_id
        self.slave = slave
        self.running = False
        self.thread = None
        self.data = UWBData()
        self.mode = mode
        self.pub_uwb = None
        self.find_working_port()

    def find_working_port(self):
        """尝试找到可用的 ttyUSB 系列串口并初始化 Modbus 客户端"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "ttyUSB" in port.device:  # 只搜索 ttyUSB 系列的串口
                try:
                    client = ModbusSerialClient(
                        method='rtu',
                        port=port.device,  # 动态选择端口
                        baudrate=115200,
                        timeout=3,
                        parity='N',
                        stopbits=1,
                        bytesize=8
                    )
                    if client.connect():  # 测试连接
                        print(f"成功连接到串口: {port.device}")
                        self.client = client
                        self.port = port.device
                        return
                except Exception as e:
                    print(f"尝试连接 {port.device} 失败: {e}")
        raise Exception("未找到可用的 ttyUSB 串口设备")

    def run(self):
        rospy.init_node(f"uwb_locator_{self.tag_id}", anonymous=True)  # 创建 ROS 节点
        self.pub_uwb = rospy.Publisher(f"/{self.tag_id}/uwb_pose", PoseStamped, queue_size=10)  # 初始化发布器
        rate = rospy.Rate(30)  # 设置发布频率为 30Hz

        if not self.client or not self.client.connect():
            print("无法连接到串口")
            return
        if self.mode == 'ANCHOR':
            # 设置为持续监测，问询式获取
            self.client.write_registers(0x003B, [0x0002], slave=self.slave)
        time.sleep(0.1)
        # 接受回复
        self.running = True
        self.thread = threading.Thread(target=self.read_serial)
        self.thread.start()

        # 发布话题
        while not rospy.is_shutdown() and self.running:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()  # 添加时间戳
            pose_stamped.header.frame_id = "uwb_frame"  # 指定坐标系名称

            # 填充位置信息
            pose_stamped.pose.position.x = self.state.pos[0]
            pose_stamped.pose.position.y = self.state.pos[1]
            pose_stamped.pose.position.z = self.state.pos[2]

            # 填充姿态信息 (全部设为 0.0)
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = 0.0
            pose_stamped.pose.orientation.w = 0.0

            self.pub_uwb.publish(pose_stamped)  # 发布到 ROS 话题
            rate.sleep()

    def read_serial(self):
        # 定时查询
        while self.running:
            buffer = None
            if self.mode == 'ANCHOR':
                start = 0x0100 + self.tag_id * 0x0016  # tag_id号标签的寄存器起始地址
                buffer = self.client.read_holding_registers(start, 0x0015, slave=self.slave)
                self.data = rtls_data_recv(buffer.registers, self.mode)
                time.sleep(0.5)
            elif self.mode == 'TAG':
                # 清理buffer
                buffer = read_modbus_frame(self.client, 51)
                self.data = rtls_data_recv(buffer, self.mode)
                time.sleep(0.5)
            print(f'tag{self.tag_id} state:{self.data}')
            self.state.pos = np.array([self.data.x, self.data.y, self.data.z])

    def stop(self):
        self.running = False
        if self.client:
            self.client.write_registers(0x003B, [0x0000], slave=self.slave)  # 停止监测
            self.thread.join()
            self.client.close()

    def __call__(self, *args, **kwargs):
        return self.state

#
# class UWBLocator_ROS:
#     def __init__(self, tag_id, slave, mode='ANCHOR'):
#         self.state = State()
#         self.client = None
#         self.port = None  # 最终选定的串口
#         self.tag_id = tag_id
#         self.slave = slave
#         self.running = False
#         self.thread = None
#         self.data = UWBData()
#         self.mode = mode
#         self.pub_uwb = None
#         self.find_working_port()
#
#     def find_working_port(self):
#         """尝试找到可用的 ttyUSB 系列串口并初始化 Modbus 客户端"""
#         ports = serial.tools.list_ports.comports()
#         for port in ports:
#             if "ttyUSB" in port.device:  # 只搜索 ttyUSB 系列的串口
#                 try:
#                     client = ModbusSerialClient(
#                         method='rtu',
#                         port=port.device,  # 动态选择端口
#                         baudrate=115200,
#                         timeout=3,
#                         parity='N',
#                         stopbits=1,
#                         bytesize=8
#                     )
#                     if client.connect():  # 测试连接
#                         print(f"成功连接到串口: {port.device}")
#                         self.client = client
#                         self.port = port.device
#                         return
#                 except Exception as e:
#                     print(f"尝试连接 {port.device} 失败: {e}")
#         raise Exception("未找到可用的 ttyUSB 串口设备")
#
#     def run(self):
#         rospy.init_node(f"uwb_locator_{self.tag_id}", anonymous=True)  # 创建 ROS 节点
#         self.pub_uwb = rospy.Publisher(f"/{self.tag_id}/uwb_pose", PoseStamped, queue_size=10)  # 初始化发布器
#         rate = rospy.Rate(30)  # 设置发布频率为 30Hz
#
#         # 捕获 SIGINT 信号
#         signal.signal(signal.SIGINT, self.signal_handler)
#
#         if not self.client or not self.client.connect():
#             print("无法连接到串口")
#             return
#         if self.mode == 'ANCHOR':
#             # 设置为持续监测，问询式获取
#             self.client.write_registers(0x003B, [0x0002], slave=self.slave)
#         time.sleep(0.1)
#         # 接受回复
#         self.running = True
#         self.thread = threading.Thread(target=self.read_serial)
#         self.thread.start()
#
#         # 发布话题
#         while not rospy.is_shutdown() and self.running:
#             pose_stamped = PoseStamped()
#             pose_stamped.header.stamp = rospy.Time.now()  # 添加时间戳
#             pose_stamped.header.frame_id = "uwb_frame"  # 指定坐标系名称
#
#             # 填充位置信息
#             pose_stamped.pose.position.x = self.state.pos[0]
#             pose_stamped.pose.position.y = self.state.pos[1]
#             pose_stamped.pose.position.z = self.state.pos[2]
#
#             # 填充姿态信息 (全部设为 0.0)
#             pose_stamped.pose.orientation.x = 0.0
#             pose_stamped.pose.orientation.y = 0.0
#             pose_stamped.pose.orientation.z = 0.0
#             pose_stamped.pose.orientation.w = 0.0
#
#             self.pub_uwb.publish(pose_stamped)  # 发布到 ROS 话题
#             rate.sleep()
#
#     def read_serial(self):
#         # 定时查询
#         while self.running:
#             buffer = None
#             if self.mode == 'ANCHOR':
#                 start = 0x0100 + self.tag_id * 0x0016  # tag_id号标签的寄存器起始地址
#                 buffer = self.client.read_holding_registers(start, 0x0015, slave=self.slave)
#                 self.data = rtls_data_recv(buffer.registers, self.mode)
#                 time.sleep(0.5)
#             elif self.mode == 'TAG':
#                 # 清理buffer
#                 buffer = read_modbus_frame(self.client, 51)
#                 self.data = rtls_data_recv(buffer, self.mode)
#                 # time.sleep(0.5)
#             print(f'tag{self.tag_id} state:{self.data}')
#             self.state.pos = np.array([self.data.x, self.data.y, self.data.z])
#
#     def signal_handler(self, signum, frame):
#         rospy.loginfo("收到终止信号，停止 UWB 节点...")
#         self.stop()
#
#     def stop(self):
#         self.running = False
#         if self.client:
#             self.client.write_registers(0x003B, [0x0000], slave=self.slave)  # 停止监测
#             self.thread.join()
#             self.client.close()
#
#     def __call__(self, *args, **kwargs):
#         return self.state


