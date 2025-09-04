import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 本案例采用FacetGrid的分面功能来实现山脊图绘制，其中最关键的一点是对于“轴”的操作
# FacetGrid默认共用x轴刻度标签
# 对于每个子图，删除y轴刻度及标签，而将分面变量作为y轴标签显示，同时重叠子图，达到将子图变成一个个“小山脉”的作用

# 设置全局样式
sns.set_theme(
    context="paper",  # 设置上下文为 "paper"，适合学术论文
    style="whitegrid",  # 使用白色网格背景
    font_scale=1.2,  # 全局字体缩放比例
    rc={
        "axes.facecolor": (0, 0, 0, 0),
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



# Create the data
rs = np.random.RandomState(1979)   # 设置随机种子值
x = rs.randn(500)                  # 随机生成500个服从标准正态分布的随机数
g = np.tile(list("ABCDEFGHIJ"), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)   # 将g列的所有值转化为ASCII码
df["x"] += m # 这样，每一个标签对应的数据就是均值为其ASCII码，方差为0的正态分布数据

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)   # 调色盘
cls = sns.color_palette(palette = 'hls',n_colors = 10)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)
# 数据源为df，分面变量设置为'g'列

# Draw the densities in a few steps
g.map(sns.kdeplot, "x",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
# fill=True：填充密度图。
# bw_adjust=.5：调整带宽，使密度估计更平滑。
# clip_on=False：允许图形超出子图的范围。
# 'x'是绘图数据变量

g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)
# 'x'是绘图数据变量


# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
# 在每个子图的 y=0 位置添加一条参考线。
# color=None 表示使用 hue 映射的颜色。


# Define and use a simple function to label the plot in axes coordinates

def label(color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
# 此处x 和 y代表相对位置参数:x = 0代表最左侧，y = 0 代表最下侧，x = 1代表最右侧，y = 1代表最上侧

g.map(label)
# 将函数 label 应用于 FacetGrid 的每个子图。
# FacetGrid 的数据分组列（g 列的值）自动传递给 label 函数的 label 参数。color自动设置为分组的颜色
# x 作为传递参数之一，但在函数内部未使用。


# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)
# 调整子图间距，使其重叠（hspace=-.25）

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
# 移除多余的标题、刻度和轴线，简化图形外观。

plt.show()