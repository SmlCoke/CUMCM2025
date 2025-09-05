import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D工具

class sample_dots:
    '''
    theta_step_rate: 角度采样的间隔百分比，角度采样间隔 = 间隔百分比(theta_step_rate) * max_theta)(np.pi)
    radius_step_rate: 半径采样的间隔百分比，半径采样间隔 = 间隔百分比(radius_step_rate) * max_radius(7)
    height_step_rate: 高度采样的间隔百分比，高度采样间隔 = 间隔百分比(height_step_rate) * max_height(10)
    '''

    def __init__(self, 
                 theta_step_rate = 0.05,
                 radius_step_rate = 0.1,
                 height_step_rate = 0.2,
                 max_radius = 7,
                 max_height = 10):
        # 生成 角度 采样维度
        max_theta = 2 * np.pi
        # arange生成数据的最大值为max_theta，防止在2Π位置处重复生成数据点
        self.theta_samples = np.arange(0, max_theta , max_theta * theta_step_rate) 

        # 生成 半径 采样维度
        self.radius_samples = np.arange(max_radius * radius_step_rate, max_radius + radius_step_rate * max_radius, radius_step_rate * max_radius)

        # 生成 顶面圆 的采样数据点，数据点坐标为 (x, y, z) 形式
        '''
        构造方式：
        x = rcos(θ)
        y = 200 + rsin(θ)
        z = 10
        '''
        top_sample_dots = [(0, 200, 10)] # 保证数据点覆盖顶面圆心

        for theta_sample in self.theta_samples:
            for radius_sample in self.radius_samples:
                top_sample_dots.append([radius_sample * np.cos(theta_sample), 
                                        200 + radius_sample * np.sin(theta_sample),
                                        10])
        
        # 生成 高度 采样维度
        # 最大值设为 max_height 保证不会生成高度为max_height的数据点，防止与top_sample_dot中的数据点重复，带来额外计算开销
        '''
        构造方式：
        x = Rcos(θ)
        y = 200 + Rsin(θ)
        z = h
        '''
        self.height_samples = np.arange(0, max_height, max_height * height_step_rate)
        side_sample_dots = []
        
        for theta_sample in self.theta_samples:
            for height_sample in self.height_samples:
                side_sample_dots.append([max_radius * np.cos(theta_sample),
                                        200 + max_radius * np.sin(theta_sample),
                                        height_sample])
                
        self.dots = np.array(top_sample_dots + side_sample_dots)
        print("============== 采样点生成成功 ^_^ ==================")

    # 打印全体采样点信息
    def show_dots(self):
        print(self.dots)

    def plot_dots_3d(self):
        """三维可视化采样点"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        xs = self.dots[:, 0]
        ys = self.dots[:, 1]
        zs = self.dots[:, 2]
        ax.scatter(xs, ys, zs, s=20, c=zs, cmap='viridis', alpha=0.7)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Sample Dots on Cylinder Surface')
        plt.savefig("采样点.png")
        plt.show()
        

if __name__ == "__main__":
    sample_dots_0 = sample_dots()
    sample_dots_0.show_dots()
    
    # # 可视化采样点，检验是否生成正确
    # sample_dots_0.plot_dots_3d()
