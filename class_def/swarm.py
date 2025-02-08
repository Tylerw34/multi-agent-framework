import os
import signal
import subprocess
from abc import ABC, abstractmethod

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import rospy
from scipy import io
from geometry_msgs.msg import PoseStamped, TwistStamped
from matplotlib.widgets import Button

from utils import quiver_data_to_segments, pose_decode, twist_decode, generate_command_line


# 虚拟类，包含Swarm的基本定义
class Swarm(ABC):
    def __init__(self):
        self.n_node = 0  # 具有的节点数量
        self.node_name_ls = []  # 节点名称列表
        self.node_states = {'pos': np.empty((0, 3)),
                            'vel': np.empty((0, 3)),
                            'acc': np.empty((0, 3)),
                            'ang': np.empty((0, 3))}  # 节点状态字典
        # 启动子线程
        self.subprocess = {}
        rospy.init_node('vboids_swarm', anonymous=True)
        self.node_subs = {}  # 订阅器字典

    # 添加节点
    def add_node(self, node_name):
        # 订阅节点的状态
        # callback_args可以用tuple传递更多参数 --- 在这里可以传递每个node的value等信息，但不是打包在pose里，而是在callback_args里
        self.node_subs[f'{node_name}_pose'] = rospy.Subscriber(f'{node_name}_pose', PoseStamped,
                                                               self.update_state, callback_args=node_name)
        self.node_subs[f'{node_name}_twist'] = rospy.Subscriber(f'{node_name}_twist', TwistStamped,
                                                                self.update_state, callback_args=node_name)
        self.n_node += 1
        self.node_name_ls.append(node_name)
        for key in self.node_states.keys():
            self.node_states[key] = np.vstack((self.node_states[key], np.zeros(3)))

    # 删除节点
    def delete_node(self, node_name):
        # 注销订阅器
        self.node_subs[f'{node_name}_pose'].unregister()
        self.node_subs.pop(f'{node_name}_pose')
        self.node_subs[f'{node_name}_twist'].unregister()
        self.node_subs.pop(f'{node_name}_twist')
        for key in self.node_states.keys():
            idx = self.node_name_ls.index(node_name)
            self.node_states[key] = np.delete(self.node_states[key], idx, axis=0)
        self.n_node -= 1
        os.kill(self.subprocess[node_name].pid, signal.SIGTERM)
        self.subprocess.pop(node_name)
        self.node_name_ls.remove(node_name)

    def update_state(self, data, node_name):
        idx = self.node_name_ls.index(node_name)
        if isinstance(data, PoseStamped):
            self.node_states['pos'][idx], self.node_states['ang'][idx] = pose_decode(data)
        elif isinstance(data, TwistStamped):
            self.node_states['vel'][idx] = twist_decode(data)


# 加入了基本绘图的基础类
class StdSwarm(Swarm):
    def __init__(self, nodes_info_dict):
        super(StdSwarm, self).__init__()
        # 获取所有节点信息
        self.node_info_dict = nodes_info_dict
        self.init_nodes()
        # 创建动画
        self.fig = plt.figure()
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot(),
                                           interval=50, cache_frame_data=False)
        plt.show()

    # 如何用命令启动节点
    @abstractmethod
    def init_nodes(self):
        pass

    def init_plot(self):
        # 创建3D图形
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter = self.ax.scatter(self.node_states['pos'][:, 0],
                                       self.node_states['pos'][:, 1],
                                       self.node_states['pos'][:, 2],
                                       marker='o', label='Position')
        self.quiver = self.ax.quiver(self.node_states['pos'][:, 0],
                                     self.node_states['pos'][:, 1],
                                     self.node_states['pos'][:, 2],
                                     self.node_states['vel'][:, 0],
                                     self.node_states['vel'][:, 1],
                                     self.node_states['vel'][:, 2],
                                     normalize=True, color='b',
                                     label='Velocity', length=0.5)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_zlim(0, 10)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.quit_button_ax = self.fig.add_axes([0.8, 0.01, 0.1, 0.05])
        self.quit_button = Button(self.quit_button_ax, 'Quit')
        self.quit_button.on_clicked(self.quit)

    def update_plot(self, frame):
        self.scatter._offsets3d = (self.node_states['pos'][:, 0],
                                   self.node_states['pos'][:, 1],
                                   self.node_states['pos'][:, 2])
        segments = quiver_data_to_segments(self.node_states['pos'][:, 0],
                                           self.node_states['pos'][:, 1],
                                           self.node_states['pos'][:, 2],
                                           self.node_states['vel'][:, 0],
                                           self.node_states['vel'][:, 1],
                                           self.node_states['vel'][:, 2])
        self.quiver.set_segments(segments)
        return self.scatter, self.quiver

    def quit(self, event):
        plt.close()
        name_ls = list(self.node_name_ls)
        for name in name_ls:
            self.delete_node(name)
        rospy.signal_shutdown('Quit')
        quit()

class BoidsSwarm(StdSwarm):
    def __init__(self, nodes_info_dict, params_style="Virtual"):
        self.params_style = params_style
        super(BoidsSwarm, self).__init__(nodes_info_dict)


    def init_nodes(self):
        name_all = list(self.node_info_dict.keys())
        for name in name_all:
            node_type = self.node_info_dict[name][0]
            ip = self.node_info_dict[name][1]
            port = self.node_info_dict[name][2]
            verbose = self.node_info_dict[name][3]
            args_dict = {'node_name': name,
                         'node_type': node_type,
                         'ip': ip,
                         'port': port,
                         'node_name_ls': name_all,
                         'params_style': self.params_style,
                         'verbose': verbose}
            interpreter = "/home/tyler/.virtualenvs/motion-capture-swarm/bin/python"
            command = generate_command_line(interpreter, args_dict)
            self.subprocess[name] = subprocess.Popen(command)
            self.add_node(name)

    def init_plot(self):
        super(BoidsSwarm, self).init_plot()
        if self.params_style == "Virtual":
            self.center = np.array([50, 50, 50])
            # 固定坐标轴范围
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(0, 100)
            self.ax.set_zlim(0, 100)
        elif self.params_style == "Motion_Capture":
            self.center = np.array([1, 1, 0])
            # 固定坐标轴范围
            self.ax.set_xlim(0, 2)
            self.ax.set_ylim(0, 2)
            self.ax.set_zlim(0, 1)
        # 用较大的星号标记中心
        self.ax.scatter(self.center[0], self.center[1], self.center[2], marker='*', s=100, c='r', label='Center')
