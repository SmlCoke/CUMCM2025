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


# Load the example planets dataset
planets = sns.load_dataset("planets")

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

g = sns.relplot(
    data=planets,
    x="distance", y="orbital_period",
    hue="year", size="mass",
    palette=cmap, sizes=(10, 200),
)

g.set(xscale="log", yscale="log")
# 使用对数坐标轴，便于显示数据分布范围较大的变量（如距离和轨道周期）。


g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
# 添加次级网格线以提升阅读性。

# g.despine(left=True, bottom=True)
plt.savefig('散点图升级2.pdf')
plt.show()

# 散点图中：
# 点的颜色由: "year"控制，变化由palette控制
# 点的大小由: "size"控制，变化由sizes控制
# 这样，可以将四组变量呈现在散点图中，分别是两组数值变量(x和y)，以及两组分类变量(hue和size)