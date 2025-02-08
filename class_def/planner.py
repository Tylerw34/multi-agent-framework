import numpy as np


def find_neighbors(pos_ls, idx, visible_range):
    """
    :param pos_ls: 位置列表
    :param idx: 当前节点的索引
    :param visible_range: 判断为邻居的距离范围
    :return: 邻居列表，平方距离列表
    """
    # 计算当前个体到所有其他个体的平方距离
    distances_squared = np.sum((pos_ls - pos_ls[idx, :]) ** 2, axis=1)
    # 确定在可见范围内的邻居，排除当前个体自身
    is_neighbor = (distances_squared > 0) & (distances_squared < visible_range ** 2)
    neighbors = np.where(is_neighbor)[0]
    return neighbors, distances_squared


def alignment(vel_ls, idx, neighbors, factor):
    """
    :param vel_ls: 速度列表
    :param idx: 当前节点的索引
    :param neighbors: 邻居列表
    :param factor: 调整因子
    :return: 对齐速度向量
    """
    if len(neighbors) == 0:
        v_alignment = np.array([0, 0, 0])
    else:
        # 计算邻居的平均速度
        avg_vel = np.mean(vel_ls[neighbors, :], axis=0)
        # 调整速度朝向平均速度
        v_alignment = (avg_vel - vel_ls[idx, :]) * factor  # 调整因子
    return v_alignment


def cohesion(pos_ls, idx, neighbors, factor):
    """
    :param pos_ls: 位置列表
    :param idx: 当前节点的索引
    :param neighbors: 邻居列表
    :param factor: 调整因子
    :return: 聚合速度向量
    """
    if len(neighbors) == 0:
        v_cohesion = np.array([0, 0, 0])
    else:
        # 计算邻居的质心
        center_of_mass = np.mean(pos_ls[neighbors, :], axis=0)
        # 朝向质心
        v_cohesion = (center_of_mass - pos_ls[idx, :]) * factor  # 调整因子
    return v_cohesion


def separation(pos_ls, idx, separation_range, factor):
    """
    :param pos_ls: 位置列表
    :param idx: 当前节点的索引
    :param separation_range: 判断为过于靠近的距离
    :param factor: 调整因子
    :return: 分离速度向量
    """
    # 根据分离距离找到邻居
    neighbors, distances_squared = find_neighbors(pos_ls, idx, separation_range)

    if len(neighbors) == 0:
        return np.array([0, 0, 0])

    # 计算距离小于分离距离的邻居的位移向量
    distances = np.sqrt(distances_squared[neighbors])
    displacements = pos_ls[idx, :] - pos_ls[neighbors, :]
    normalized_displacements = displacements / distances[:, np.newaxis]

    # 累加所有的分离向量
    v_separation = np.sum(normalized_displacements, axis=0) * factor  # 分离强度的调整因子

    return v_separation


def center_following(pos_ls, idx, distance_range, center, factor):
    """
    :param pos_ls: 位置列表
    :param idx: 当前节点的索引
    :param distance_range: 判断为离中心太远的距离
    :param center: 集群中心
    :param factor: 调整因子
    :return: 向心速度向量
    """
    # 计算个体到中心的距离
    distance_to_center = np.linalg.norm(pos_ls[idx, :] - center)
    # 判断个体是否离边界太远
    if distance_to_center > distance_range:
        # 计算一个引导个体朝向边界的向量
        v_border_following = (center - pos_ls[idx, :]) / np.linalg.norm(
            center - pos_ls[idx, :]) * factor
        # v_border_following = (center - pos_ls[idx, :]) * factor
    else:
        # 如果个体离边界足够远，则不需要动作
        v_border_following = np.array([0, 0, 0])
    return v_border_following


# not used
def smooth_heading(vel_ls, idx, max_heading_change):
    """
    :param vel_ls: 速度列表
    :param idx: 当前节点的索引
    :param max_heading_change: 最大方向变化
    :return: 平滑速度分量
    """
    # 计算方向变化
    current_vel = vel_ls[idx, :]
    previous_vel = current_vel - np.array([0.01, 0.01])  # 以前的速度，按需要调整

    # 计算当前速度和以前速度之间的角度变化
    angle_change = np.degrees(
        np.arctan2(current_vel[0] * previous_vel[1] - current_vel[1] * previous_vel[0],
                   current_vel[0] * previous_vel[0] + current_vel[1] * previous_vel[1]))

    # 检查角度变化是否超过阈值
    if angle_change > max_heading_change:
        # 逐渐调整方向以平滑变化
        target_vel = (current_vel + previous_vel) / 2
        v_smooth_heading = (target_vel - current_vel) * 0.1  # 平滑调整因子
    else:
        # 如果方向变化在阈值内，则不需要动作
        v_smooth_heading = np.array([0, 0])
    return v_smooth_heading


# Boids集群算法
class Boids:
    def __init__(self,
                 visible_range=5,
                 separation_range=10,
                 border_distance=30,
                 center=np.array([50, 50, 50]),
                 alignment_factor=0.05,
                 cohesion_factor=0.01,
                 separation_factor=0.1,
                 center_following_factor=0.1,
                 max_vel=9,
                 min_vel=3,
                 boarder_buffer=50):
        """
        :param visible_range: alignment和cohesion的邻居搜索范围
        :param separation_range: separation的邻居搜索范围
        :param border_distance: 判断为离中心太远的距离
        :param center: 集群中心
        :param alignment_factor: alignment调整因子
        :param cohesion_factor: cohesion调整因子
        :param separation_factor: separation调整因子
        :param center_following_factor: center_following调整因子
        :param max_vel: 最大速度
        :param min_vel: 最小速度
        :param boarder_buffer: 中心到边界距离
        """
        self.visible_range = visible_range
        self.separation_distance = separation_range
        self.border_distance = border_distance
        self.center = center
        self.alignment_factor = alignment_factor
        self.cohesion_factor = cohesion_factor
        self.separation_factor = separation_factor
        self.center_following_factor = center_following_factor
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.boarder_min = center + np.array([-boarder_buffer, -boarder_buffer, -boarder_buffer])
        self.boarder_max = center + np.array([boarder_buffer, boarder_buffer, boarder_buffer])

    def __call__(self, pos_ls, vel_ls, idx):
        """
        :param pos_ls: 所有节点的位置列表
        :param vel_ls: 所有节点的速度列表
        :param idx: 当前节点序号
        :return: 更新后的速度列表
        """
        # 计算邻居
        neighbors, _ = find_neighbors(pos_ls, idx, self.visible_range)
        # 计算各个速度分量
        v_alignment = alignment(vel_ls, idx, neighbors, self.alignment_factor)
        v_cohesion = cohesion(pos_ls, idx, neighbors, self.cohesion_factor)
        v_separation = separation(pos_ls, idx, self.separation_distance, self.separation_factor)
        v_center_following = center_following(pos_ls, idx, self.border_distance, self.center,
                                              self.center_following_factor)
        # 更新速度
        new_vel = vel_ls[idx, :] + (v_alignment + v_cohesion + v_separation + v_center_following)
        # 限制速度，保留正负号
        speed = np.linalg.norm(new_vel)
        if speed > self.max_vel:
            new_vel = new_vel / speed * self.max_vel
        elif speed < self.min_vel:
            new_vel = new_vel / speed * self.min_vel

        # 到达边界区时反向
        pos_this = pos_ls[idx, :]
        new_vel = np.where(pos_this < self.boarder_min, np.abs(new_vel), new_vel)
        new_vel = np.where(pos_this > self.boarder_max, -np.abs(new_vel), new_vel)
        return new_vel

class BearingOnlyStatic:
    def __init__(self, target_bearing_dict):
        self.names = list(target_bearing_dict.keys())
        neighbor_num = len(self.names)
        dim = len(list(target_bearing_dict.values())[0])
        self.pos = np.zeros(dim)
        self.target_bearing_ls = np.zeros((neighbor_num, 3))
        self.other_pos_ls = np.zeros((neighbor_num, dim))
        for i, name in enumerate(self.names):
            self.target_bearing_ls[i] = target_bearing_dict[name] / np.linalg.norm(target_bearing_dict[name])

    def __call__(self, self_state, other_states):
        self.pos = self_state.pos
        for i, name in enumerate(self.names):
            self.other_pos_ls[i, :] = other_states[name].pos
        target_vel = self.calculate()
        return target_vel

    def calculate(self):
        error_ls = self.other_pos_ls - self.pos
        # 单位化，得到方向向量
        norms = np.linalg.norm(error_ls, axis=1, keepdims=True)
        bearing_ls = error_ls / norms
        bearing_error = bearing_ls - self.target_bearing_ls
        # 计算目标速度
        target_vel = np.sum(bearing_error, axis=0)
        return target_vel

class BearingOnlyDynamic(BearingOnlyStatic):
    def __init__(self, target_bearing_dict, kp, ki):
        super(BearingOnlyDynamic, self).__init__(target_bearing_dict)
        self.error_sum = np.zeros(3)
        self.kp = kp
        self.ki = ki

    def calculate(self):
        bearing_error = super(BearingOnlyDynamic, self).calculate()
        self.error_sum += bearing_error
        target_vel = self.kp * bearing_error + self.ki * self.error_sum
        return target_vel

# Obstacle Avoidance Leader in Bearing Only
class BearingOnlyLOA:
    def __init__(self, obstacle_func, dist_th, speed):
        self.obstacle_func = obstacle_func
        self.dist_th = dist_th  # 距离阈值
        self.speed = speed

    def __call__(self, self_state):
        target_vel = np.array([self.speed, 0, 0])
        if self.obstacle_func is None:
            return target_vel
        dist_low = self.dist_th[0]
        dist_high = self.dist_th[1]
        self.pos = self_state.pos
        x = self.pos[0]
        y = self.pos[1]
        y_obstacle = self.obstacle_func(x)
        distance = np.abs(y - y_obstacle)
        if distance < dist_low:
            target_vel[1] = (dist_low - distance) * np.sign(y - y_obstacle)
        elif distance > dist_high:
            target_vel[1] = (dist_high - distance) * np.sign(y - y_obstacle)
        return target_vel



