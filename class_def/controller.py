import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button


# PID控制器
class PID:
    def __init__(self, name, kp=0.0, ki=0.0, kd=0.0, degree=2, verbose=False):
        """
        :param name: 控制器名称
        :param kp: 比例系数
        :param ki: 积分系数
        :param kd: 微分系数
        :param degree: 控制器阶数
        :param verbose: 是否显示图像
        """
        # 控制器参数
        self.name = name    # 控制器名称
        self.kp = kp    # 比例系数
        self.ki = ki    # 积分系数
        self.kd = kd    # 微分系数
        self.degree = degree    # 控制器阶数
        # 控制器储存的数据
        self.state = 0.0    # 状态量
        self.target = 0.0   # 目标状态量
        self.u = 0.0    # 控制量
        self.error_last = 0.0   # 上一次误差
        self.error_sum = 0.0    # 误差积分
        if verbose:     # 是否显示图像
            self.history = np.empty((0, 3))  # 存储绘图用的历史数据
            # 绘图相关变量
            self.fig, (self.ax_state, self.ax_u) = plt.subplots(2, 1)   # 创建图像
            # 标题
            self.ax_state.set_title(f'{name} PID Controller')    # 状态量标题
            self.state_curve, = self.ax_state.plot([], [], label=name, color='b')   # 状态量曲线
            self.target_curve, = self.ax_state.plot([], [], label=f'target_{name}',
                                                    color='r', linestyle='--')  # 目标量曲线
            self.u_curve, = self.ax_u.plot([], [], label=f'{name}_u', color='g')    # 控制量曲线
            # 退出按钮
            self.quit_button_ax = self.fig.add_axes([0.8, 0.01, 0.1, 0.05])
            self.quit_button = Button(self.quit_button_ax, 'Quit')
            self.quit_button.on_clicked(self.quit_plot)     # 退出按钮事件
            # 绘制动画
            self.ani = animation.FuncAnimation(self.fig, self.update_plot,
                                               interval=100, cache_frame_data=False, blit=False)

    def step(self):
        """
        :return: 返回下一步控制量
        """
        error = self.target - self.state
        self.error_sum += error
        error_diff = error - self.error_last
        self.error_last = error
        du = self.kp * error + self.ki * self.error_sum + self.kd * error_diff
        if self.degree == 2:    # 二阶控制器，控制位置
            self.u += du
        else:   # 一阶控制器，控制速度
            self.u = du
        return self.u

    def reset(self):
        """
        重置控制器
        """
        self.error_last = 0.0
        self.error_sum = 0.0

    def quit_plot(self, event):
        """
        退出按钮事件
        """
        # 保存并停止动画
        plt.savefig(f'figures/{self.name}_{self.kp}_{self.ki}_{self.kd}.png')
        self.ani.event_source.stop()
        plt.close()

    def update_plot(self, frame):
        """
        更新绘图数据
        """
        # 更新历史数据
        self.history = np.vstack((self.history, [self.state, self.target, self.u]))
        time_data = np.arange(0, len(self.history), 1)
        # 更新曲线
        self.state_curve.set_data(time_data, self.history[:, 0])
        self.target_curve.set_data(time_data, self.history[:, 1])
        self.u_curve.set_data(time_data, self.history[:, 2])
        # 根据数据更新坐标轴
        self.ax_state.relim()
        self.ax_state.autoscale_view()
        self.ax_u.relim()
        self.ax_u.autoscale_view()
        return self.ax_state, self.ax_u
