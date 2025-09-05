import numpy as np


# 计算 安全角 余弦值
def calculate_sage_angle(center,
                         radius,
                         M_coordinate):
    '''
    输入参数：
    center: 含有三个元素的numpy数组， 表示烟幕球心坐标
    radius: 烟幕球半径
    M_coordinate: 含有三个元素的numpy数组，导弹坐标

    返回结果：
    导弹对烟幕云团的视锥的锥角(safe_angle)(余弦值)，只要某一点与中心连线的夹角小于安全角，则判定为 “遮蔽”
    '''
    # 计算导弹距离云团中心的距离
    distance = np.linalg.norm(M_coordinate - center) 

    return np.sqrt(distance**2 - radius**2)/distance

# 对于给定采样点，计算 待评角 
def calculate_dot_angle(center,
                        M_coordinate,
                        dot_coordinate):
    '''
    输入参数：
    center: 烟幕弹云团中心坐标，类型为含有三个元素的numpy数组
    M_coordinate: 导弹坐标，类型为含有三个元素的numpy数组
    dot_coordinate: 采样点坐标，类型为含有三个元素的numpy数组

    返回参数：
    导弹与采样点连线 与 导弹与烟幕云团球心连线 的夹角余弦值
    '''
    vector_M_center =  center - M_coordinate

    vector_M_dot = dot_coordinate - M_coordinate

    # np.linalg.norm: 取向量模长
    # np.dot(): 取向量点积
    return np.dot(vector_M_center, vector_M_dot)/(np.linalg.norm(vector_M_center) * np.linalg.norm(vector_M_dot))


# 检测样本点是否处于遮蔽范围内，是则返回 True, 否则返回 False
def detect_dot(dot_coordinate,
               M_coordinate,
               center,
               radius):
    '''
    输入参数：
    dot_coordinate: 采样点坐标，类型为含有三个元素的numpy数组
    M_coordinate: 导弹坐标，类型为含有三个元素的numpy数组
    center: 烟幕弹云团中心坐标，类型为含有三个元素的numpy数组
    radius: 烟幕球半径
    '''
    return calculate_dot_angle(center = center, 
                               dot_coordinate = dot_coordinate, 
                               M_coordinate = M_coordinate) > calculate_sage_angle(center = center,
                                                                                   radius = radius,
                                                                                   M_coordinate = M_coordinate)


# 根据起爆点坐标，当前 scene 信息，求遮蔽时长（但烟幕云团）
def get_max_shallow_time_pro1(flash_coordinate,
                              current_scene,
                              time_step_rate,
                              sample_dots):
    # 当前无人机，导弹信息
    scene = current_scene
    
    # 计算烟幕云团最长下落时间
    max_falling_time = flash_coordinate[2] / 3

    # 计算导弹击中假目标耗时
    max_hit_time = current_scene.M_coordinates[0][0] / current_scene.M_velocity[0][0]

    # 最长考虑时间
    max_time = max_falling_time if max_hit_time > max_falling_time else max_hit_time

    # 时间点采样
    time_samples = np.arange(0, max_time + time_step_rate * max_time, time_step_rate * max_time)

    time_samples_counts = len(time_samples)

    # 不同时间节点下的导弹M1坐标



if __name__ == "__main__":
    center = np.array([0,0,0])
    radius = 1
    M_coordinate = np.array([2, 0, 0])
    dot_coordinate = np.array([-2, 1.732, 0]) # 擦边检验
    
    print(calculate_sage_angle(np.array([0,0,0]), 1, np.array([2, 0, 0])))
    
    print(calculate_dot_angle(np.array([0,0,0]),
                              np.array([0,2,0]),
                              np.array([2,0,0])))
    
    print(detect_dot(center = center,
                     radius = radius,
                     M_coordinate = M_coordinate,
                     dot_coordinate = dot_coordinate))