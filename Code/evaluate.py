import numpy as np
import copy

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

# 对于给定采样点，计算 测试角 
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
    # 计算导弹距离云团中心的距离
    M_c_distance = np.linalg.norm(M_coordinate - center) 
    # 计算导弹距离采样点的距离
    M_d_distance = np.linalg.norm(M_coordinate - dot_coordinate)

    # 计算测试角余弦值
    cos_dot_angle = calculate_dot_angle(center = center, 
                                         dot_coordinate = dot_coordinate, 
                                         M_coordinate = M_coordinate)
    
    # 计算安全角余弦值
    cos_safe_angle = calculate_sage_angle(center = center,
                                          radius = radius,
                                          M_coordinate = M_coordinate)

    return M_d_distance > M_c_distance and cos_dot_angle > cos_safe_angle

# 根据起爆点坐标，当前 scene 信息，求遮蔽时长（但烟幕云团）
def get_max_shallow_time_pro1(flash_coordinate,
                              current_scene,
                              time_step_rate,
                              sample_dots):
    # 当前无人机，导弹信息
    scene = current_scene
    
    print(f"当前系统时间: {scene.t}")
    # 计算烟幕云团最长下落时间
    max_falling_time = flash_coordinate[2] / 3
    print(f"[info][debug]: max_falling_time = {max_falling_time}")

    # 计算导弹击中假目标耗时
    max_hit_time = current_scene.M_coordinates[0][0] / np.abs((current_scene.M_velocity[0][0]))
    print(f"[info][debug]: max_hit_time = {max_hit_time}")

    # 最长考虑时间
    max_time = max_falling_time if max_hit_time > max_falling_time else max_hit_time

    max_time = max_time if max_time < 20 else 20

    print(f"[info][debug]: max_time = {max_time}")
    # 时间点采样
    time_samples = np.arange(0, max_time + time_step_rate * max_time, time_step_rate * max_time)

    time_samples_counts = len(time_samples)
    print(f"采样时间节点:\n{time_samples}")
    # 计算不同时间节点下的导弹坐标、烟幕云团中心坐标
    centers = [flash_coordinate]
    M1_coordinates = [scene.M_coordinates[0]]


    # 不同时间节点下的导弹M1坐标
    for time_sample_index, time_sample in  enumerate(time_samples[1:]):
        '''
        time_sample_index 从 0 开始
        time_sample 从 time_samples[1] 开始
        '''
        # 上次的时间节点，演化到此时的耗时
        evolve_time = time_sample - time_samples[time_sample_index]
        print(f"time_sample: {time_sample}")
        print(f"time_samples[time_sample_index - 1]: {time_samples[time_sample_index]}")
        print(f"第{time_sample_index}次演化耗时:{evolve_time}")
        # 计算系统演化到当前时间节点时的状态
        scene.load_current_state(evolve_time = evolve_time, verbose = True)

        # 计算当前时间节点下的导弹坐标
        M1_coordinates.append(scene.M_coordinates[0]) 

        # 计算当前时间节点下的烟幕云团坐标
        current_center = copy.deepcopy(centers[time_sample_index])
        current_center[2] = current_center[2] - 3 * evolve_time

        centers.append(current_center)
    
    # 节点个数如果不等于时间节点个数，强制报错退出
    node_counts = len(centers) 
    if (node_counts != time_samples_counts):
        raise ValueError("严重错误：节点个数如果不等于时间节点个数")
    
    shallow_flags = []
    print(f"centers: {centers}")
    for index in range(node_counts):
        shallow_flag = True
        print(f"在第{index + 1}个时间点，导弹坐标: {M1_coordinates[index]}")
        print(f"在第{index + 1}个时间点，烟幕云团中心坐标: {centers[index]}")
        for dot_index, dot in enumerate(sample_dots):
            if detect_dot(dot, M_coordinate = M1_coordinates[index], center = centers[index], radius = 10):
                print(f"采样点{dot_index + 1}:{sample_dots[dot_index].tolist()}在第{index + 1}个时间节点遮蔽成功")
                # continue
            else:
                shallow_flag = False
                print(f"采样点{dot_index + 1}:{sample_dots[dot_index].tolist()}在第{index + 1}个时间节点遮蔽失败")
                # break
        shallow_flags.append(shallow_flag)

    print(f"\n采样时间节点:\n{time_samples}")
    print(f"\n不同时间节点的真目标遮蔽状态(True表示成功遮蔽，False表示遮蔽失败)：\n{shallow_flags}")



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