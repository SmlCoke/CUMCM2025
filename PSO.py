import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
S_14 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 14)  # 宋体

class ParticleSwarmOptimization:
    def __init__(self, 
                 objective_func, 
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
                 mutation_rate=0.1):  # 变异率
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
        """
        self.objective_func = objective_func
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
        
        # 初始化粒子位置和速度
        self.positions = np.zeros((n_particles, n_dimensions))
        self.velocities = np.zeros((n_particles, n_dimensions))
        
        # 初始化个体最优和全局最优
        self.pbest_positions = np.zeros((n_particles, n_dimensions))
        self.pbest_values = np.ones(n_particles) * float('inf')
        self.gbest_position = np.zeros(n_dimensions)
        self.gbest_value = float('inf')
        
        # 初始化历史记录，用于跟踪算法性能
        self.history = {
            'gbest_value': [],
            'gbest_position': []
        }
        
    def initialize_swarm(self):
        """初始化所有粒子的位置和速度"""
        # 在每个维度的边界内随机初始化位置
        for i in range(self.n_dimensions):
            min_bound, max_bound = self.bounds[i]
            self.positions[:, i] = np.random.uniform(min_bound, max_bound, self.n_particles)
            # 初始化速度为位置范围的一定比例
            self.velocities[:, i] = np.random.uniform(
                -(max_bound - min_bound) * 0.1, 
                (max_bound - min_bound) * 0.1, 
                self.n_particles
            )
        
        # 评估初始位置
        for i in range(self.n_particles):
            value = self.objective_func(self.positions[i])
            
            # 更新个体最优
            self.pbest_positions[i] = self.positions[i].copy()
            self.pbest_values[i] = value
            
            # 更新全局最优
            if value < self.gbest_value:
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
        
        # 限制速度大小
        for i in range(self.n_dimensions):
            min_bound, max_bound = self.bounds[i]
            max_velocity = 0.2 * (max_bound - min_bound)  # 增大了最大速度限制
            self.velocities[:, i] = np.clip(self.velocities[:, i], -max_velocity, max_velocity)
    
    def update_positions(self):
        """更新所有粒子的位置并确保在边界内"""
        # 更新位置
        self.positions += self.velocities
        
        # 确保所有粒子位置在搜索空间内
        for i in range(self.n_dimensions):
            min_bound, max_bound = self.bounds[i]
            self.positions[:, i] = np.clip(self.positions[:, i], min_bound, max_bound)
        
        # 应用随机扰动（变异）以帮助跳出局部最优
        self.apply_mutation()
    
    def apply_mutation(self):
        """对部分粒子进行随机扰动（变异）"""
        # 对每个粒子，以一定概率进行变异
        for i in range(self.n_particles):
            if np.random.random() < self.mutation_rate:
                # 随机选择一个维度进行变异
                dim = np.random.randint(0, self.n_dimensions)
                min_bound, max_bound = self.bounds[dim]
                
                # 变异强度随迭代减弱
                mutation_strength = 0.1 * (max_bound - min_bound)
                
                # 添加高斯噪声
                self.positions[i, dim] += np.random.normal(0, mutation_strength)
                
                # 确保变异后的位置仍在边界内
                self.positions[i, dim] = np.clip(self.positions[i, dim], min_bound, max_bound)
    
    def evaluate_swarm(self):
        """评估所有粒子并更新最优位置"""
        for i in range(self.n_particles):
            value = self.objective_func(self.positions[i])
            
            # 更新个体最优
            if value < self.pbest_values[i]:
                self.pbest_positions[i] = self.positions[i].copy()
                self.pbest_values[i] = value
                
                # 更新全局最优
                if value < self.gbest_value:
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
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['gbest_value'], '-o', markersize=0)
        plt.title('PSO 算法收敛曲线', fontproperties = S_14)
        plt.xlabel('迭代次数', fontproperties = S_14)
        plt.ylabel('目标函数值', fontproperties = S_14)
        plt.grid(True)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()