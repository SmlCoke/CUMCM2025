import numpy as np
import random
import math
import time
import copy
from matrix2mux_tree import get_and_count

def simulated_annealing_optimize(sel_vars, data_matrix, max_time=300, 
                                initial_temp=100.0, cooling_rate=0.95, min_temp=0.1):
    """
    使用模拟退火算法优化变量顺序以最小化AND门数量
    
    参数:
        sel_vars: 选择信号列表，如 ["sel[0]", "sel[1]", ...]，以这个列表中的顺序开始退火
        data_matrix: 数据矩阵，numpy数组
        max_time: 最大运行时间(秒)，默认5分钟
        initial_temp: 初始温度
        cooling_rate: 冷却率
        min_temp: 最小温度
        
    返回:
        best_order: 找到的最佳变量顺序
        best_count: 最佳顺序下的AND门数量
    """
    # 获取变量基础名称和范围
    var_base = sel_vars[0].split('[')[0]  # 将sel_vars[0]字符串按照"["拆分为两项，第一项就是信号名字
    var_indices = [int(v.split('[')[1].split(']')[0]) for v in sel_vars]  # 提取索引
    # int(v.split('[')[1]代表：i]
    # "i]".split(']')[0]代表：i
    
    # 使用正向顺序作为起点 (n, n-1, ..., 1, 0)
    n = max(var_indices) + 1  # n就是选择信号位宽
    current_indices = var_indices # 初始顺序（索引），即正向顺序
    current_order = sel_vars # 初始顺序（变量），即正向顺序
    
    # print(f"初始变量顺序: {current_order}")
    
    # 计算初始AND门数量
    current_count = get_and_count(current_order, data_matrix)
    # print(f"初始AND门数量: {current_count}")
    
    # 记录最佳解
    best_order = current_order.copy()
    best_count = current_count
    
    # 模拟退火参数：当前温度
    temp = initial_temp   
    
    # 记录不接受新解的连续次数
    no_accept_count = 0
    max_no_accept = 100  # 连续100次不接受新解则重启
    
    # 开始计时
    start_time = time.time()
    iteration = 0
    
    # 记录每次改进
    improvements = []
    
    # 主循环
    while temp > min_temp and time.time() - start_time < max_time:
        iteration += 1
        
        # 生成新的候选解（两种扰动策略的组合）
        neighbor_order = generate_neighbor(current_order)
        neighbor_indices = [int(v.split('[')[1].split(']')[0]) for v in neighbor_order] # 打乱后的索引，用于构建新的数据矩阵
        data_matrix_new = reorder_data_matrix(data_matrix, neighbor_indices)
        # 计算新解的AND门数量
        try:
            neighbor_count = get_and_count(neighbor_order, data_matrix_new)
            
            # 计算变化量，如果小于零则代表该顺序更优
            delta = neighbor_count - current_count
            
            # 接受准则：Metropolis准则
            if delta < 0:  # 更好的解，肯定接受
                current_order = neighbor_order
                current_count = neighbor_count
                no_accept_count = 0
                
                # 更新最佳解
                if current_count < best_count:
                    best_order = current_order.copy()
                    best_count = current_count
                    improvements.append((iteration, best_count))
                    # print(f"[改进] 迭代 {iteration}, 温度 {temp:.2f}, AND门个数: {best_count}")
            elif random.random() < math.exp(-delta / temp):  # 以一定概率接受较差解，温度越低，接受概率越小
                current_order = neighbor_order
                current_count = neighbor_count
                no_accept_count = 0
            else:
                no_accept_count += 1
        except Exception as e:
            print(f"计算过程出错: {e}")
            no_accept_count += 1
        
        # 降温，当迭代次数为10的整数倍时，执行降温
        if iteration % 10 == 0:
            temp = cooling_rate * temp
        
        # 如果长时间没有接受新解，考虑重启
        if no_accept_count >= max_no_accept:
            # print(f"[重启] 连续 {max_no_accept} 次未接受新解，进行随机重启")
            # 随机打乱顺序作为新起点
            random_indices = current_indices.copy()
            random.shuffle(random_indices)  # 打乱顺序
            current_order = [f"{var_base}[{idx}]" for idx in random_indices]
            current_indices = [int(v.split('[')[1].split(']')[0]) for v in current_order]
            data_matrix_new = reorder_data_matrix(data_matrix, current_indices)
            current_count = get_and_count(current_order, data_matrix_new)
            no_accept_count = 0
            
            # 稍微提高温度允许更多探索
            temp = max(temp * 1.5, initial_temp * 0.5)
        
        # 定期输出状态
        if iteration % 50 == 0:
            elapsed = time.time() - start_time
            # print(f"迭代 {iteration}, 已用时间 {elapsed:.1f}秒, 温度 {temp:.2f}, 当前解: {current_count}, 最佳解: {best_count}")
    
    # 算法结束
    total_time = time.time() - start_time
    # print(f"\n模拟退火完成:")
    # print(f"总迭代次数: {iteration}")
    # print(f"总用时: {total_time:.1f}秒")
    # print(f"最佳变量顺序: {best_order}")
    # print(f"最佳AND门数量: {best_count}")
    
    '''
    # 输出改进历史
    if improvements:
        print("\n改进历史:")
        for it, count in improvements:
            print(f"迭代 {it}: {count}")
    '''
    return best_order, best_count

def generate_neighbor(current_order):
    """
    生成当前顺序的邻居解
    
    使用两种扰动策略:
    1. 随机交换两个变量位置 (概率0.7)
    2. 随机选择一个变量并移动到随机位置 (概率0.3)
    """
    neighbor = current_order.copy()
    
    if random.random() < 0.7:
        # 策略1: 随机交换两个变量
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    else:
        # 策略2: 随机移动一个变量
        old_pos = random.randint(0, len(neighbor) - 1)
        new_pos = random.randint(0, len(neighbor) - 1)
        if old_pos != new_pos:
            var = neighbor.pop(old_pos)
            neighbor.insert(new_pos, var)
    
    return neighbor

def multi_start_annealing(sel_vars, data_matrix, num_starts=3, time_per_start=120):
    """
    多起点模拟退火优化
    
    参数:
        sel_vars: 选择信号列表
        data_matrix: 数据矩阵
        num_starts: 起点数量
        time_per_start: 每个起点的运行时间(秒)
        
    返回:
        overall_best_order: 所有起点中找到的最佳顺序
        overall_best_count: 最佳顺序的AND门数量
    """
    print(f"执行{num_starts}起点模拟退火优化，每个起点最多运行{time_per_start}秒")
    
    var_base = sel_vars[0].split('[')[0]
    var_indices = [int(v.split('[')[1].split(']')[0]) for v in sel_vars]
    n = max(var_indices) + 1
    
    # 定义起点顺序
    starting_points = []
    
    # 起点1: 正向顺序 (n-1, n-2, ..., 1, 0)
    forward_indices = list(range(n-1, -1, -1))
    starting_points.append([f"{var_base}[{idx}]" for idx in forward_indices])
    forward_and_count = get_and_count(starting_points[-1], reorder_data_matrix(data_matrix, forward_indices))
    print("正向顺序的AND门个数: ", forward_and_count)

    # 起点2: 反向顺序 (0, 1, 2, ..., n-1)
    reverse_indices = list(range(0, n))
    starting_points.append([f"{var_base}[{idx}]" for idx in reverse_indices])
    reverse_and_count = get_and_count(starting_points[-1], reorder_data_matrix(data_matrix, reverse_indices))
    print("逆向顺序的AND门个数: ", reverse_and_count)

    
    # 额外起点: 随机顺序
    for _ in range(num_starts - 2):
        random_indices = list(range(n))
        random.shuffle(random_indices)  # 随机打乱
        starting_points.append([f"{var_base}[{idx}]" for idx in random_indices])
    
    # 初始化最优顺序
    if (reverse_and_count < forward_and_count):
        overall_best_order = starting_points[1]  # 全局最优顺序
        overall_best_count = reverse_and_count # 全局最优And门个数
    else:
        overall_best_order = starting_points[0]  # 全局最优顺序
        overall_best_count = forward_and_count # 全局最优And门个数
    
    # 对每个起点执行模拟退火
    for i, start_order in enumerate(starting_points):
        print(f"\n===== 起点 {i+1}/{num_starts} =====")
        print(f"起点顺序: {start_order}")
        
        # 为当前起点构建一个临时的sel_vars列表
        temp_sel_vars = start_order.copy()
        temp_sel_indices = [int(v.split('[')[1].split(']')[0]) for v in temp_sel_vars]

        # 生成新的输出矩阵
        data_matrix_new = reorder_data_matrix(data_matrix, temp_sel_indices)
        # 执行模拟退火
        best_order, best_count = simulated_annealing_optimize(
            temp_sel_vars, data_matrix_new, 
            max_time=time_per_start,  # 每个起点退火的时间限制
            initial_temp=100.0,
            cooling_rate=0.95,
            min_temp=0.1
        )
        
        # 更新全局最佳解
        if best_count < overall_best_count:
            overall_best_count = best_count
            overall_best_order = best_order.copy()
    
    print("\n===== 多起点优化完成 =====")
    print(f"全局最佳变量顺序: {overall_best_order}")
    print(f"全局最佳AND门数量: {overall_best_count}")
    
    return overall_best_order, overall_best_count


# 在新的sel顺序下，重排列data_matrix
def reorder_data_matrix(data_matrix, new_order):
    n = len(new_order)
    num_rows = data_matrix.shape[0]
    assert 2 ** n == num_rows, "行数不是2的幂，无法对应n位选择信号"
    assert sorted(new_order) == list(range(n)), "new_order必须是0到n-1的排列"

    # 0. 在索引意义下，sel物理顺序与索引顺序之和为 n - 1
    for i in range(n):
        new_order[i] = n - 1 - new_order[i]

    # 1. 生成原始地址矩阵（每行一个地址，形如[0,0,1,...]）
    addr_matrix = np.array([list(np.binary_repr(i, width=n)) for i in range(num_rows)], dtype=int)

    # 2. 根据 new_order 打乱列顺序
    scrambled_addr_matrix = addr_matrix[:, new_order]

    # 3. 将每行（地址）视为一个二进制数，计算其对应的十进制值
    powers_of_two = 2 ** np.arange(n-1, -1, -1)  
    # 权重：从最高位到最低位: [n-1, n-2, n-3, ..., 0]
    scrambled_addr_values = scrambled_addr_matrix.dot(powers_of_two) # 矩阵乘法

    # 4. 获取排序后的地址索引（对应新地址从小到大）
    reorder_indices = np.argsort(scrambled_addr_values)

    # 5. 用这个索引重排 data_matrix
    reordered_data_matrix = data_matrix[reorder_indices]

    return reordered_data_matrix


# 示例使用
if __name__ == "__main__":
    # 构造示例数据矩阵，这里使用与原代码相同的示例
    data = np.array([
        [0, -1, -1, -1],  # 地址 000
        [-1, 1, -1, -1],  # 地址 001
        [-1, -1, 0, -1],  # 地址 010
        [-1, -1, -1, 0],  # 地址 011
        [-1, -1, -1, -1],  # 地址 100
        [-1, -1, 1, -1],  # 地址 101
        [-1, -1, -1, -1],  # 地址 110
        [-1, -1, -1, -1],  # 地址 111
    ])
    
    sel_vars = ["sel[2]", "sel[1]", "sel[0]"]
    
    print(reorder_data_matrix(data, [2,1,0]))

    # 执行多起点模拟退火
    best_order, best_count = multi_start_annealing(sel_vars, data, num_starts=10, time_per_start=60)

