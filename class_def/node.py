import signal
import socket
import threading
import time
from abc import ABC, abstractmethod
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from matplotlib import pyplot as plt

from controller import PID
from planner import Boids, BearingOnlyStatic, BearingOnlyDynamic, BearingOnlyLOA
from locator import MotionCapture
from utils import State, state_encode, twist_decode, pose_decode, cvt_ctrl_to_car_ctrl, get_pid_state_from_node_state


# 节点类
class Node(ABC):
    def __init__(self, name, other_names, rate=30):
        """
        初始化节点
        :param name: 节点名称
        :param other_names: 其他节点名称列表
        :param rate: ros节点发布频率
        """
        # 数据存储部分
        self.name = name  # 节点名称
        if name in other_names:
            other_names.remove(name)  # 去除自己的名字
        self.other_names = other_names  # 储存其他节点名称
        self.state = State()  # 节点状态
        self.other_states = {}  # 储存其他节点的状态
        for other_name in self.other_names:
            self.other_states[other_name] = State()
        self.subs = {}  # 储存对其他节点的订阅

        # 节点处理相关
        signal.signal(signal.SIGTERM, self.clean_up)  # 绑定关闭信号到清理函数
        self.stop_flag = threading.Event()  # 线程停止标志
        self.pub_thread = threading.Thread(target=self.pub_loop)  # 线程：定时发布状态
        # ROS相关
        rospy.init_node(self.name, anonymous=False)  # 初始化ROS节点
        self.rate = rospy.Rate(rate)  # 状态发布频率
        self.pub_pose = rospy.Publisher(name + '_pose', PoseStamped, queue_size=10)  # 发布位置/角度状态
        self.pub_twist = rospy.Publisher(name + '_twist', TwistStamped, queue_size=10)  # 发布速度状态

    # 主循环
    def pub_loop(self):
        while not self.stop_flag.is_set() and not rospy.is_shutdown():
            self.update_from_locator()  # 更新状态
            self.update_from_planner()  # 更新目标状态
            self.publish()  # 发布状态
            self.rate.sleep()  # 等待下一次发布

    # 发布自身状态
    def publish(self):
        pose_msg, twist_msg = state_encode(self.state)  # 将状态转换为消息
        self.pub_pose.publish(pose_msg)  # 发布位置/角度状态
        self.pub_twist.publish(twist_msg)  # 发布速度状态

    # 从定位器更新自身状态
    @abstractmethod
    def update_from_locator(self):
        pass

    # 从规划器更新目标状态
    @abstractmethod
    def update_from_planner(self):
        pass

    # 订阅回调函数
    @abstractmethod
    def subscriber(self, data, name):
        data_type = data.__class__
        if data_type == PoseStamped:
            return pose_decode(data)  # 解码位置/角度
        elif data_type == TwistStamped:
            return twist_decode(data)  # 解码速度

    # 启动节点
    @abstractmethod
    def run(self):
        self.pub_thread.start()
        for name in self.other_names:
            self.subs[name + '_pose'] = rospy.Subscriber(name + '_pose', PoseStamped, self.subscriber,
                                                         callback_args=name)  # 开始订阅其他节点的位置/角度状态
            self.subs[name + '_twist'] = rospy.Subscriber(name + '_twist', TwistStamped, self.subscriber,
                                                          callback_args=name)  # 开始订阅其他节点的速度状态

    # 关闭信号回调函数，清理资源
    def clean_up(self, signum, frame):
        self.__del__()

    # 析构函数
    def __del__(self):
        self.stop_flag.set()  # 设置线程停止标志
        self.pub_thread.join()  # 等待发布线程结束
        for sub in self.subs.values():
            sub.unregister()
        rospy.signal_shutdown("Node shutdown\n")
        print(f"{self.name} is terminated\n")
        quit()  # 退出进程


# 虚拟boids节点
class VBoidsNode(Node):
    def __init__(self, name, other_names=None, params_style="Virtual"):
        """
        初始化虚拟boids节点
        :param name: 节点名称
        :param other_names: 其他节点名称列表
        :param params_style: boids参数类型
        """
        self.center = np.array([0.5, 0.5, 0])
        if params_style == "Virtual":
            super(VBoidsNode, self).__init__(name, other_names)
            self.state.pos = (np.random.rand(3)) * 50
            self.state.vel = (np.random.rand(3)) * 5
            self.boids = Boids()
        elif params_style == "Motion_Capture":
            super(VBoidsNode, self).__init__(name, other_names, rate=30)  # 降低频率
            self.state.pos = (np.random.rand(3)) * 1  # 场地更小
            self.state.vel = (np.random.rand(3)) * 0.15
            # 固定在水平面内
            self.state.vel[2] = 0
            self.state.pos[2] = 0
            self.boids = Boids(visible_range=0.1,  # 调整boids参数
                               separation_range=0.3,
                               border_distance=0.3,
                               center=self.center,
                               alignment_factor=0.05,
                               cohesion_factor=0.01,
                               separation_factor=0.1,
                               center_following_factor=0.01,
                               max_vel=0.2,
                               min_vel=0.05,
                               boarder_buffer=0.8)
        # 储存其他节点的位置和速度，方便计算
        self.pos_ls = np.zeros((len(other_names) + 1, 3))  # 所有节点的位置列表，包括自己（在最后）
        self.vel_ls = np.zeros((len(other_names) + 1, 3))  # 所有节点的速度列表，包括自己（在最后）
        # 初始化节点
        self.state.acc = np.zeros(3)  # 视作质点，不需要加速度
        self.state.ang = np.zeros(3)  # 视作质点，不需要朝向角度
        self.target_vel = np.zeros(3)  # 当前目标速度

    # 订阅回调函数（override）
    def subscriber(self, data, name):
        data_decoded = super(VBoidsNode, self).subscriber(data, name)  # 调用父类的订阅函数，解码数据
        data_type = data.__class__
        if data_type == PoseStamped:
            self.other_states[name].pos, self.other_states[name].ang = data_decoded
            self.pos_ls[self.other_names.index(name), :] = self.other_states[name].pos  # 更新其他节点的位置
        elif data_type == TwistStamped:
            self.other_states[name].vel = data_decoded
            self.vel_ls[self.other_names.index(name), :] = self.other_states[name].vel  # 更新其他节点的速度

    # 从定位器更新自身状态（override）
    def update_from_locator(self):
        self.state.vel = self.target_vel  # 更新速度（直接赋值）
        self.state.pos += self.state.vel * self.rate.sleep_dur.to_sec()  # 更新位置（直接赋值）
        self.vel_ls[-1, :] = self.state.vel  # 更新boids中自己的速度
        self.pos_ls[-1, :] = self.state.pos  # 更新boids中自己的位置

    # 从规划器更新目标状态（override）
    def update_from_planner(self):
        self.target_vel = self.boids(self.pos_ls, self.vel_ls, -1)  # 根据boids算法更新目标速度

    def run(self):
        # 启动线程
        super(VBoidsNode, self).run()
        input("Press Enter to terminate...\n")  # 等待终止
        self.__del__()  # 析构


# 动捕boids节点
class BoidsNode(VBoidsNode):
    def __init__(self, name, ip, port, other_names=None, verbose=0):
        """
        初始化动捕boids节点
        :param name: 节点名称
        :param ip: 小车控制通信ip地址
        :param port: 小车控制通信端口号
        :param other_names: 其他节点名称列表
        :param verbose: 是否显示图像
        """
        # 初始化节点
        super(BoidsNode, self).__init__(name, other_names, params_style="Motion_Capture")
        # 修改参数
        self.center = np.array([1, 1, 0])
        self.state = State()  # 节点状态初始为0
        self.boids = Boids(visible_range=0.4,  # 调整boids参数
                           separation_range=0.3,
                           border_distance=0.5,
                           center=self.center,
                           alignment_factor=0.05,
                           cohesion_factor=0.01,
                           separation_factor=0.1,
                           center_following_factor=0.01,
                           max_vel=0.2,
                           min_vel=0.05,
                           boarder_buffer=0.8)
        # 动捕定位器
        self.locator = MotionCapture(name)
        # 控制相关
        self.verbose = verbose  # 是否显示控制图像
        self.speed_pid = PID("speed", kp=30, kd=10, verbose=self.verbose)  # 径向速度PID控制器
        self.angle_pid = PID("angle", kp=30, kd=2, degree=1, verbose=self.verbose)  # 角度PID控制器
        self.ctrl_flag = True  # 是否需要实际发送控制指令
        self.ctrl_period = self.rate.sleep_dur.to_sec() / 5  # 控制周期
        self.ctrl_thread = threading.Thread(target=self.ctrl_loop)  # 控制线程
        # socket通讯相关
        self.ip = ip
        self.port = int(port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 从定位器更新自身状态（override）
    def update_from_locator(self):
        # 更新状态
        self.state = self.locator()
        # 更新boids中自己的位置和速度
        self.pos_ls[-1, :] = self.state.pos  # 更新boids中自己的位置
        if np.linalg.norm(self.state.vel) < 0.02:
            self.vel_ls[-1, :] = (self.center - self.state.pos) * 0.02  # 如果速度太低，用朝向中心的速度更新boids
        else:
            self.vel_ls[-1, :] = self.state.vel  # 更新boids中自己的速度

    def send_ctrl(self, speed, angle):
        """
        发送控制指令
        :param speed: 径向速度
        :param angle: 角速度
        :return:
        """
        buffer = cvt_ctrl_to_car_ctrl(speed, angle)  # 将控制输入转换为小车控制输入
        command = "<%d,%d,%d,%d>" % (buffer[0], buffer[1], buffer[2], buffer[3])
        self.socket.sendto(command.encode(), (self.ip, self.port))  # 发送控制指令

    # PID反馈控制循环
    def ctrl_loop(self):
        while not self.stop_flag.is_set():
            state = self.locator()  # 获取最新状态
            self.speed_pid.state, self.speed_pid.target, self.angle_pid.state, self.angle_pid.target \
                = get_pid_state_from_node_state(state, self.target_vel)  # 更新PID控制器
            if self.ctrl_flag:
                speed_u = self.speed_pid.step()  # 计算径向速度控制量
                angle_u = self.angle_pid.step()  # 计算角度控制量
                self.send_ctrl(speed_u, angle_u)  # 发送控制指令
            time.sleep(self.ctrl_period)  # 等待下一次控制
        self.send_ctrl(0, 0)  # 发送停止指令
        self.socket.close()  # 关闭socket

    def run(self):
        Node.run(self)
        # 启动控制线程
        self.locator.run()
        self.ctrl_thread.start()
        # 显示控制图像（可选）
        if self.verbose:
            plt.show()
        input("Press Enter to terminate...\n")  # 等待终止
        self.__del__()  # 析构

    # 析构（override）
    def __del__(self):
        super(BoidsNode, self).__del__()
        self.angle_pid.quit_plot(None)  # 退出角度控制器绘图
        self.speed_pid.quit_plot(None)  # 退出速度控制器绘图
        self.locator.stop()
        self.ctrl_thread.join()  # 结束线程


if __name__ == '__main__':
    import argparse
    import json

    node = None
    parser = argparse.ArgumentParser(description='get std args')
    parser.add_argument('--node_name', type=str, help='Node name')
    parser.add_argument('--node_type', type=str, help='Node type')
    args, remaining = parser.parse_known_args()
    _node_name = args.node_name
    _node_type = args.node_type
    if _node_type == 'VBoidsNode' or _node_type == 'BoidsNode':
        parser = argparse.ArgumentParser(description='get boids args')
        parser.add_argument('--node_name_ls', type=str, help='Node name list (JSON)')
        parser.add_argument('--ip', type=str, help='IP address')
        parser.add_argument('--port', type=str, help='Port number')
        parser.add_argument('--params_style', type=str, help='params_style')
        parser.add_argument('--verbose', type=int, help='Verbose')
        args = parser.parse_args(remaining)
        _node_name_ls = json.loads(args.node_name_ls)
        _ip = args.ip
        _port = args.port
        _params_style = args.params_style
        _verbose = args.verbose
        if _node_type == 'VBoidsNode':
            node = VBoidsNode(_node_name, _node_name_ls, params_style=_params_style)
        else:
            node = BoidsNode(_node_name, _ip, _port, _node_name_ls, verbose=int(_verbose))

    node.run()

