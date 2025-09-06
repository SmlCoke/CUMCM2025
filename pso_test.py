import numpy as np
from Code.PSO import ParticleSwarmOptimization

# 示例评估函数：Sphere 函数（一个简单的测试函数）
def sphere_function(x):
    """球面函数，全局最小值为 f(0,0,...,0) = 0"""
    return np.sum(x**2)

# 示例评估函数：Rosenbrock 函数
def rosenbrock_function(x):
    """Rosenbrock 函数，全局最小值为 f(1,1,...,1) = 0"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# 示例评估函数：Rastrigin 函数
def rastrigin_function(x):
    """Rastrigin 函数，全局最小值为 f(0,0,...,0) = 0"""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def main():
    # 定义问题参数
    n_dimensions = 5  # 变量数量
    bounds = [(-5.12, 5.12) for _ in range(n_dimensions)]  # Rastrigin函数的标准边界
    
    # 创建 PSO 实例，使用 Rastrigin 函数作为测试
    pso = ParticleSwarmOptimization(
        objective_func=rastrigin_function,
        n_dimensions=n_dimensions,
        bounds=bounds,
        n_particles=100,  # 增加粒子数量
        max_iter=200,     # 增加最大迭代次数
        w_start=0.9,      # 初始惯性权重较大，促进全局搜索
        w_end=0.4,        # 最终惯性权重较小，促进局部搜索
        c1=2.0,
        c2=2.0,
        stagnation_limit=20,  # 连续20次无改进才考虑停止
        mutation_rate=0.1     # 10%的变异率
    )
    
    # 运行优化，强制完成所有迭代
    best_position, best_value, history = pso.optimize(verbose=True, force_full_iter=True)
    
    # 绘制收敛曲线
    pso.plot_convergence()

if __name__ == "__main__":
    main()