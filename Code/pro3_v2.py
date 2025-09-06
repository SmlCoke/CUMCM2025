import numpy as np
import copy 
from scene import scene, get_Flashpoint
from evaluate import get_max_shallow_time_pro1, detec_scene_bomb, count_true_segments
from sample_dots import sample_dots
# from matplotlib.font_manager import FontProperties


class ParticleSwarmOptimization:
    def __init__(self, 
                 n_dimensions,  # 变量的维度
                 bounds,        # 每个维度的取值范围
                 n_particles=50,  # 粒子数量
                 w_start=0.9,     # 初始惯性权重
                 w_end=0.4,       # 最终惯性权重
                 c1=2.0,          # 认知参数(个体学习因子)
                 c2=2.0,          # 社会参数(群体学习因子)
                 max_iter=100,    # 最大迭代次数
                 min_err=1e-6,    # 最小误差阈值
                 stagnation_limit=15, # 停滞计数器限制
                 mutation_rate=0.1, # 变异率
                 scene = None,           # 当前场景
                 sample_dots = None):    # 采样点
        """
        初始化粒子群优化算法
        
        参数:
            objective_func: 目标函数(适应度函数)，用于评估每个粒子的位置
            n_dimensions: 搜索空间的维度数(变量数量)
            bounds: 每个维度的上下界，格式为[(min1, max1), (min2, max2), ...]
            n_particles: 粒子数量
            w_start: 初始惯性权重
            w_end: 最终惯性权重
            c1: 认知参数(个体学习因子)
            c2: 社会参数(群体学习因子)
            max_iter: 最大迭代次数
            min_err: 最小误差阈值
            stagnation_limit: 停滞计数器限制，连续多少次没有改进才考虑提前终止
            mutation_rate: 变异率，用于随机扰动粒子
            scene: 当前导弹/无人机状态
            sample_dot: 采样点，numpy数组
        """
        self.n_dimensions = n_dimensions
        self.n_particles = n_particles
        self.bounds = bounds
        self.w_start = w_start
        self.w_end = w_end
        self.w = w_start  # 当前惯性权重
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.min_err = min_err
        self.stagnation_limit = stagnation_limit
        self.mutation_rate = mutation_rate
        self.current_scene = copy.deepcopy(scene)
        self.sample_dots = copy.deepcopy(sample_dots)
        
        # 初始化粒子位置和速度
        self.positions = np.zeros((n_particles, n_dimensions))
        self.velocities = np.zeros((n_particles, n_dimensions))
        
        # 初始化个体最优和全局最优
        self.pbest_positions = np.zeros((n_particles, n_dimensions))
        self.pbest_values = np.zeros(n_particles)
        self.gbest_position = np.zeros(n_dimensions)
        self.gbest_value = 0
        
        # 初始化历史记录，用于跟踪算法性能
        self.history = {
            'gbest_value': [],
            'gbest_position': []
        }
        
    def initialize_swarm(self):
        """初始化所有粒子的位置和速度"""
        # 初始化决策变量的前两维：速度方向和速度大小
        for i in range(2):
            min_bound, max_bound = self.bounds[i]
            self.positions[:, i] = np.random.uniform(min_bound, max_bound, self.n_particles)
            # 初始化速度为位置范围的一定比例
            self.velocities[:, i] = np.random.uniform(
                -(max_bound - min_bound) * 0.1, 
                (max_bound - min_bound) * 0.1, 
                self.n_particles
            )
        
        # 初始化最大演化时间（导弹M1击中假目标时，系统停止演化）
        T = np.sqrt(20000**2 + 2000**2) / 300
        
        # 初始化决策变量的后六维：三个投放时间和三个引爆时间
        
        # 初始化三个投放时间决策变量
        # print(f"sel_positions: {self.positions}")
        for particle_index in range(len(self.positions)):
            # 初始化第一次投放时间变量对应的位置
            self.positions[particle_index][2] = np.random.uniform(0, 5)
            # 初始化第一次投放时间变量对应的速度
            A = 5
            self.velocities[particle_index][2] = np.random.uniform(-0.1 * T, 0.1 * T)
            
            # 初始化第二次投放时间变量对应的位置
            self.positions[particle_index][3] = np.random.uniform(self.positions[particle_index][2] + 1, self.positions[particle_index][2]  + 6)
            # 初始化第二次投放时间变量对应的速度
            A = 5
            self.velocities[particle_index][3] = np.random.uniform(-0.1 * A, 0.1 * A)
            
            # 初始化第三次投放时间变量对应的位置
            self.positions[particle_index][4] = np.random.uniform(self.positions[particle_index][3] + 1, self.positions[particle_index][3] + 6)
            # 初始化第三次投放时间变量对应的速度
            A = 5
            self.velocities[particle_index][4] = np.random.uniform(-0.1 * A, 0.1 * A)

        # 初始化三个引爆时间决策变量
        for particle_index in range(len(self.positions)):
            # 初始化第一次引爆时间变量对应的位置
            self.positions[particle_index][5] = np.random.uniform(self.positions[particle_index][2], self.positions[particle_index][2] + 5)
            # 初始化第一次引爆时间变量对应的速度
            A = 5
            self.velocities[particle_index][5] = np.random.uniform(-0.1 * A, 0.1 * A)
            
            # 初始化第二次引爆时间变量对应的位置
            self.positions[particle_index][6] = np.random.uniform(self.positions[particle_index][3], self.positions[particle_index][3] + 5)
            # 初始化第二次引爆时间变量对应的速度
            A = 5
            self.velocities[particle_index][6] = np.random.uniform(-0.1 * A, 0.1 * A)
            
            # 初始化第三次引爆时间变量对应的位置
            self.positions[particle_index][7] = np.random.uniform(self.positions[particle_index][4], self.positions[particle_index][4] + 5)
            # 初始化第三次引爆时间变量对应的速度
            A = 5
            self.velocities[particle_index][7] = np.random.uniform(-0.1 * A, 0.1 * A)

        # 矫正初始决策变量，确保Pro2中的决策变量能够被包含
        for i in range(self.n_particles):
            if np.random.uniform(0,1) < 0.3:
                self.positions[0] = 3.1331701
                self.positions[1] = 129.34776284
                self.positions[2] = 1
                self.positions[3] = 5.12309655
        
        # 评估初始位置
        for i in range(self.n_particles):
            value = self.get_max_shallow_time_pro3(*self.positions[i])
            
            # 更新个体最优
            self.pbest_positions[i] = self.positions[i].copy()
            self.pbest_values[i] = value
            
            # 更新全局最优
            if value > self.gbest_value:
                self.gbest_value = value
                self.gbest_position = self.positions[i].copy()
        
        # 记录初始状态
        self.history['gbest_value'].append(self.gbest_value)
        self.history['gbest_position'].append(self.gbest_position.copy())
    
    def update_velocities(self, iter):
        """
        更新所有粒子的速度
        
        参数:
            iter: 当前迭代次数，用于更新惯性权重
        """
        # 更新惯性权重 (线性递减)
        self.w = self.w_start - (self.w_start - self.w_end) * iter / self.max_iter
        
        # 生成随机因子
        r1 = np.random.random((self.n_particles, self.n_dimensions))
        r2 = np.random.random((self.n_particles, self.n_dimensions))
        
        # 计算认知部分和社会部分
        cognitive_component = self.c1 * r1 * (self.pbest_positions - self.positions)
        social_component = self.c2 * r2 * (self.gbest_position - self.positions)
        
        # 更新速度
        self.velocities = self.w * self.velocities + cognitive_component + social_component
        
        # 初始化最大演化时间（导弹M1击中假目标时，系统停止演化）
        T = np.sqrt(20000**2 + 2000**2) / 300

        # 限制速度大小(前两维)
        for i in range(2):
            min_bound, max_bound = self.bounds[i]
            max_velocity = 0.2 * (max_bound - min_bound)  # 增大了最大速度限制
            self.velocities[:, i] = np.clip(self.velocities[:, i], -max_velocity, max_velocity)

        # 限制速度大小(后六维)
        for particle_index in range(len(self.positions)):
            A_deli_1 = 5
            max_velocity = 0.2 * A_deli_1
            self.velocities[particle_index][2] = np.clip(self.velocities[particle_index][2], -max_velocity, max_velocity)

            A_deli_2 = 5
            max_velocity = 0.2 * A_deli_2
            self.velocities[particle_index][3] = np.clip(self.velocities[particle_index][3], -max_velocity, max_velocity)

            A_deli_3 = 5
            max_velocity = 0.2 * A_deli_3
            self.velocities[particle_index][4] = np.clip(self.velocities[particle_index][4], -max_velocity, max_velocity)

            A_bomb_1 = 5
            max_velocity = 0.2 * A_bomb_1
            self.velocities[particle_index][5] = np.clip(self.velocities[particle_index][5], -max_velocity, max_velocity)

            A_bomb_2 = 5
            max_velocity = 0.2 * A_bomb_2
            self.velocities[particle_index][6] = np.clip(self.velocities[particle_index][6], -max_velocity, max_velocity)

            A_bomb_3 = 5
            max_velocity = 0.2 * A_bomb_3
            self.velocities[particle_index][7] = np.clip(self.velocities[particle_index][7], -max_velocity, max_velocity)
    
    def update_positions(self):
        """更新所有粒子的位置并确保在边界内"""
        # 更新位置
        self.positions += self.velocities
        
        # 初始化最大演化时间（导弹M1击中假目标时，系统停止演化）
        T = np.sqrt(20000**2 + 2000**2) / 300

        # 确保所有粒子位置在搜索空间内
        for i in range(2):
            min_bound, max_bound = self.bounds[i]
            self.positions[:, i] = np.clip(self.positions[:, i], min_bound, max_bound)

        for particle_index in range(len(self.positions)):
            self.positions[particle_index][2] = np.clip(self.positions[particle_index][2], 
                                                        0, 
                                                        5)

            self.positions[particle_index][3] = np.clip(self.positions[particle_index][3], 
                                                        self.positions[particle_index][2] + 1,
                                                        self.positions[particle_index][2] + 6)
            
            self.positions[particle_index][4] = np.clip(self.positions[particle_index][4], 
                                                        self.positions[particle_index][3] + 1,
                                                        self.positions[particle_index][3] + 6)

            self.positions[particle_index][5] = np.clip(self.positions[particle_index][5], 
                                                        self.positions[particle_index][2],
                                                        self.positions[particle_index][2] + 5)
            
            self.positions[particle_index][6] = np.clip(self.positions[particle_index][6], 
                                                        self.positions[particle_index][3],
                                                        self.positions[particle_index][3] + 5)
            
            self.positions[particle_index][7] = np.clip(self.positions[particle_index][7], 
                                                        self.positions[particle_index][4],
                                                        self.positions[particle_index][4] + 5)
        # 应用随机扰动（变异）以帮助跳出局部最优
        self.apply_mutation()
    
    def apply_mutation(self):
        """对部分粒子进行随机扰动（变异）"""
        # 对每个粒子，以一定概率进行变异
        for i in range(self.n_particles):
            if np.random.random() < self.mutation_rate:
                self.positions[i] = self.gen_new_particle()
    
    # 产生一个新粒子，用于变异函数
    def gen_new_particle(self):
        position = np.zeros(self.n_dimensions)
        # 初始化决策变量的前两维：速度方向和速度大小
        for i in range(2):
            min_bound, max_bound = self.bounds[i]
            position[i] = np.random.uniform(min_bound, max_bound)
        
        # 初始化最大演化时间（导弹M1击中假目标时，系统停止演化）
        T = np.sqrt(20000**2 + 2000**2) / 300
        
        # 初始化决策变量的后六维：三个投放时间和三个引爆时间
        # 初始化三个投放时间决策变量
        # 初始化第一次投放时间变量对应的位置
        position[2] = np.random.uniform(0, 5)
        # 初始化第二次投放时间变量对应的位置
        position[3] = np.random.uniform(position[2] + 1, position[2] + 6)
        # 初始化第三次投放时间变量对应的位置
        position[4] = np.random.uniform(position[3] + 1, position[3] + 6)
            

        # 初始化三个引爆时间决策变量
        # 初始化第一次引爆时间变量对应的位置
        position[5] = np.random.uniform(position[2], position[2] + 5)
        # 初始化第二次引爆时间变量对应的位置
        position[6] = np.random.uniform(position[3], position[3] + 5)
        # 初始化第三次引爆时间变量对应的位置
        position[7] = np.random.uniform(position[4], position[4] + 5)

        return position
            
        
    def evaluate_swarm(self):
        """评估所有粒子并更新最优位置"""
        for i in range(self.n_particles):
            value = self.get_max_shallow_time_pro3(*self.positions[i])
            
            # 更新个体最优
            if value > self.pbest_values[i]:
                self.pbest_positions[i] = self.positions[i].copy()
                self.pbest_values[i] = value
                
                # 更新全局最优
                if value > self.gbest_value:
                    self.gbest_value = value
                    self.gbest_position = self.positions[i].copy()
        
        # 记录当前状态
        self.history['gbest_value'].append(self.gbest_value)
        self.history['gbest_position'].append(self.gbest_position.copy())
    
    def optimize(self, verbose=True, force_full_iter=False):
        """
        运行PSO优化过程
        
        参数:
            verbose: 是否打印优化进度
            force_full_iter: 是否强制完成所有迭代，不提前终止
            
        返回:
            tuple: (最优位置, 最优值, 历史记录)
        """
        # 初始化粒子群
        self.initialize_swarm()
        
        stagnation_counter = 0
        prev_gbest = self.gbest_value
        
        # 打印初始最优值
        if verbose:
            print(f"初始最优值: {self.gbest_value}")
        
        # 迭代直到达到最大迭代次数
        for iter in range(self.max_iter):
            # 更新速度
            self.update_velocities(iter)
            
            # 更新位置
            self.update_positions()
            
            # 评估粒子
            self.evaluate_swarm()
            
            # 检查是否达到收敛条件
            if abs(self.gbest_value - prev_gbest) < self.min_err:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # 仅当允许提前终止且连续多次没有改进时才考虑停止
            if not force_full_iter and stagnation_counter >= self.stagnation_limit:
                if verbose:
                    print(f"连续 {stagnation_counter} 次迭代无显著改进，于迭代 {iter + 1}/{self.max_iter} 停止")
                    print(f"最优值: {self.gbest_value}")
                break
            
            prev_gbest = self.gbest_value
            
            # 打印进度
            if verbose:
                print(f"迭代 {iter + 1}/{self.max_iter}, 最优值: {self.gbest_value}")
            else:
                if (iter + 1) % 10 == 0:
                    print(f"迭代 {iter + 1}/{self.max_iter}, 最优值: {self.gbest_value}")
        
        if verbose:
            print(f"优化完成! 最终最优值: {self.gbest_value}")
            print(f"最优位置: {self.gbest_position}")
        
        return self.gbest_position, self.gbest_value, self.history
    
    # def plot_convergence(self):
    #     """绘制收敛曲线"""
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(self.history['gbest_value'], '-o', markersize=0)
    #     plt.title('PSO 算法收敛曲线', fontproperties = S_14)
    #     plt.xlabel('迭代次数', fontproperties = S_14)
    #     plt.ylabel('目标函数值', fontproperties = S_14)
    #     plt.grid(True)
    #     plt.yscale('log')
    #     plt.tight_layout()
    #     plt.show()
    
    # 根据粒子的实际状态，得到该系统在这种粒子信息情况下推演到最后的遮蔽总时间
    def get_max_shallow_time_pro3(self,
                                  velocity_direction,
                                  velocity_magnitude,
                                  deli_time_1,                                  
                                  deli_time_2,
                                  deli_time_3,
                                  bomb_time_1,
                                  bomb_time_2,                                  
                                  bomb_time_3):
        '''
        输入参数：
        velocity_direction: 无人机速度方向，numpy数组
        velocity_magnitude: 无人机速度大小，numpy数组
        deli_time_i: 无人机投放烟幕弹i的时刻
        bomb_time_i: 烟幕弹i起爆的时刻
        '''
        scene_pos = copy.deepcopy(self.current_scene)

        # 根据方位角和速度大小，获取FY1的速度矢量
        velocity = np.zeros((5, 3))
        velocity[0][0] = np.cos(velocity_direction)*velocity_magnitude
        velocity[0][1] = np.sin(velocity_direction)*velocity_magnitude
        
        # 将当前粒子的决策变量信息加载进入系统状态scene_pos
        scene_pos.FY_velocity = copy.deepcopy(velocity)

        # 最长考虑时间，对这个时间段进行采样 
        max_time = np.sqrt(20000**2 + 2000**2)/300
        # 采样时间间隔百分比
        time_step_rate = 0.01
        
        time_samples = np.arange(0, max_time + time_step_rate * max_time, time_step_rate * max_time)
        
        shallow_flags = []
        # print(f"shallow_flags: {shallow_flags}")
        for time_sample_index, time_sample in enumerate(time_samples):
            shallow_flag_1 = detec_scene_bomb(scene = scene_pos,
                                              deli_time = deli_time_1,
                                              bomb_time = bomb_time_1,
                                              current_time = time_sample,
                                              sample_dots = self.sample_dots)
            
            shallow_flag_2 = detec_scene_bomb(scene = scene_pos,
                                              deli_time = deli_time_2,
                                              bomb_time = bomb_time_2,
                                              current_time = time_sample,
                                              sample_dots = self.sample_dots)
            
            shallow_flag_3 = detec_scene_bomb(scene = scene_pos,
                                              deli_time = deli_time_3,
                                              bomb_time = bomb_time_3,
                                              current_time = time_sample,
                                              sample_dots = self.sample_dots)
            
            # 三个烟幕云团只要有一个满足遮蔽条件，则判定为当前时刻真目标被完全遮蔽
            # print(f"time_sample_index = {time_sample_index}")
            # print(f"shallow_flag_1 = {shallow_flag_1}")
            # print(f"shallow_flag_2 = {shallow_flag_2}")
            # print(f"shallow_flag_3 = {shallow_flag_3}")
            shallow_flags.append(shallow_flag_1 or shallow_flag_2 or shallow_flag_3)

        shallow_list = np.array(count_true_segments(shallow_flags))

        # 最长遮蔽时间 = Σ(时间节点间隔 * 第i个True序列的True个数 - 1)
        return max_time * time_step_rate * np.sum(shallow_list - 1)
            

        
if __name__ == "__main__":
    # 无人机初始位置坐标
    FY_coordinates_ini = np.array([[17800, 0, 1800],
                                   [12000, 1400, 1400],
                                   [6000, -3000, 700],
                                   [11000, 2000, 1800],
                                   [13000, -2000, 1300]])
    # 导弹初始位置坐标
    M_coordinates_ini = np.array([[20000, 0, 2000],
                                  [19000, 600, 2100],
                                  [18000, -600, 1900]])
    
    # 无人机初始速度
    FY_velocity_ini = np.zeros((5, 3))
    #  print(type(FY_velocity_ini))
    # 加载系统状态
    scene_test_pro3 = scene(FY_coordinates_ini, M_coordinates_ini, FY_velocity_ini, True)
    
    # 加载采样点
    sample_dots_pro3 = copy.deepcopy(sample_dots().dots)

    # 定义问题参数
    n_dimensions = 8  # 变量数量
    bounds = [(np.pi - np.arccos(89/np.sqrt(7922)), np.pi),  # 决策变量一——速度方向范围
              (70, 140),     # 决策变量二——速度大小范围
              (0, np.sqrt(20000**2 + 2000**2)/300 / 3), # 决策变量三——飞行至投放点的时间范围              
              (1, np.sqrt(20000**2 + 2000**2)/300 / 3), # 决策变量四——飞行至投放点的时间范围
              (1, np.sqrt(20000**2 + 2000**2)/300 / 3), # 决策变量五——飞行至投放点的时间范围
              (0,5),   # 决策变量六——投放至起爆时间范围，这三个变量的上界由flight1 + 2 + 3 以及 导弹击中假目标耗时 限定
              (0,5),   # 决策变量七——投放至起爆时间范围，这三个变量的上界由flight1 + 2 + 3 以及 导弹击中假目标耗时 限定
              (0,5)    # 决策变量八——投放至起爆时间范围，这三个变量的上界由flight1 + 2 + 3 以及 导弹击中假目标耗时 限定
              ] 
    

    print(f"参数范围:\n {bounds}")
    # 创建 PSO 实例，使用 Rastrigin 函数作为测试
    pso = ParticleSwarmOptimization(
        n_dimensions=n_dimensions,
        bounds=bounds,
        n_particles=20,  # 增加粒子数量
        max_iter=50,     # 增加最大迭代次数
        w_start=0.9,      # 初始惯性权重较大，促进全局搜索
        w_end=0.4,        # 最终惯性权重较小，促进局部搜索
        c1=2.0,
        c2=2.0,
        stagnation_limit=50,  # 连续20次无改进才考虑停止
        mutation_rate=0.4,     # 10%的变异率
        scene = scene_test_pro3,
        sample_dots = sample_dots_pro3
    )
    
    pso.optimize()     

    print(f"测试是否正确? {pso.get_max_shallow_time_pro3(*pso.gbest_position)}")
    print(f"测试是否正确? {pso.get_max_shallow_time_pro3(*pso.gbest_position)}")            