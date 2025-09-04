import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(
    context="paper",  # 设置上下文为 "paper"，适合学术论文
    style="whitegrid",  # 使用白色网格背景
    font_scale=1.2,  # 全局字体缩放比例
    rc={
        "font.family": "serif",  # 使用衬线字体（如 Times New Roman）
        "font.serif": ["Times New Roman"],  # 指定衬线字体
        "axes.labelsize": 12,  # 轴标签字体大小
        "xtick.labelsize": 10,  # x轴刻度字体大小
        "ytick.labelsize": 10,  # y轴刻度字体大小
        "legend.fontsize": 10,  # 图例字体大小
        "axes.titlesize": 14,  # 图标题字体大小
        "figure.titlesize": 16,  # 整体图的标题字体大小
        "grid.linewidth": 0,  # 网格线宽度
        "lines.linewidth": 1.5,  # 线条宽度
        "axes.linewidth": 0.8,  # 坐标轴线宽度
        "patch.linewidth": 0.8,  # 图形边框宽度
    }
)

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")

cls = sns.color_palette(palette = 'hls',n_colors = 2)

# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri,palette = cls,ci = 95)
plt.show()


# 注：lineplot:
'''
lineplot 的参数
以下是一些关键参数的作用：

参数	功能
data	    数据集，支持 Pandas DataFrame 或 NumPy 数组。
x 和 y	    指定要绘制的变量，x 通常是独立变量，y 是因变量。
hue	        用于分组数据，给每个组分配不同的颜色。
style	    用于区分线型样式（虚线、实线等）。
ci	        设置置信区间，例如 ci=95 绘制 95% 的误差带，ci=None 则不绘制。
estimator	聚合函数，用于计算误差带范围的统计量（如均值、标准差）。
palette	    颜色调色板，用于指定趋势线的颜色集合。
linewidth	设置线条宽度。

'''