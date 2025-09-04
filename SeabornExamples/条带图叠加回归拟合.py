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
        "axes.linewidth": 1.2,  # 坐标轴线宽度
        "patch.linewidth": 0.8,  # 图形边框宽度
    }
)

mpg = sns.load_dataset("mpg")
sns.catplot(
    data=mpg, x="cylinders", y="acceleration", hue="weight",
    native_scale=True, zorder=1
)
sns.regplot(
    data=mpg, x="cylinders", y="acceleration",
    scatter=False, truncate=False, order=2, color=".2",
)
plt.show()

# 参数：
# sns.catplot():
#   data=mpg：数据源为 mpg 数据集。
#   x: 分类变量。
#   y: 连续变量。
#   hue="weight"：通过颜色深浅表示汽车重量。
#   native_scale=True：保留原始数据比例。
#   zorder=1：设置绘图层次，确保数据点在低层次绘制（较低的 z-order 值）。
# sns.regplot()
#   scatter=False：关闭散点图，仅绘制拟合曲线。
#   truncate=False：曲线不截断，覆盖完整的横轴范围。
#   order=2：指定拟合曲线为二阶多项式。
#   color=".2"：设置拟合曲线颜色为浅灰色（.2 表示灰度值）。
#   线性回归注意事项：
#   (1) 将分类变量x视为数值变量
#   (2) 虽然每一个x对应多个y，但不会对y取均值或中位数，而是所有数据点共同参与拟合，求出误差平方和最小化