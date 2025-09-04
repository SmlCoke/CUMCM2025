import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import colormaps as cm
from matplotlib.colors import LinearSegmentedColormap


# 设置主题
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

# 创建数据
rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list("ABCDEFGHIJ"), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)
df["x"] += m

# 初始化 FacetGrid
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)   # 调色盘
cls = sns.color_palette(palette = 'hls',n_colors = 10)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)


# 自定义渐变色填充函数
def gradient_fill_density(data, color, label):
    # 核密度估计
    kde = sns.kdeplot(data["x"], bw_adjust=0.5, clip_on=False, fill=False, alpha=1, linewidth=1.5)

    # 获取密度曲线的 x 和 y 数据
    x_vals = kde.get_lines()[0].get_xdata()
    y_vals = kde.get_lines()[0].get_ydata()

    # 创建渐变背景
    img_data = np.linspace(0, 1, len(x_vals)).reshape(1, -1)  # 渐变色梯度
    cmap = cm.viridis
    extent = [x_vals.min(), x_vals.max(), 0, y_vals.max()]  # 背景覆盖范围
    ax = plt.gca()
    im = ax.imshow(img_data, aspect="auto", cmap=cmap, extent=extent)

    # 绘制密度曲线并裁剪渐变背景
    fill_line, = ax.fill(x_vals, y_vals, facecolor="none", edgecolor="none")
    ax.plot(x_vals, y_vals, color="k", lw=1.5)  # 绘制边界线
    im.set_clip_path(fill_line)  # 裁剪背景


# 应用自定义填充到每个子图
g.map_dataframe(gradient_fill_density)

def label(color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
# 此处x和y代表相对位置参数:x = 0代表最左侧，y = 0 代表最下侧，x = 1代表最右侧，y = 1代表最上侧

g.map(label)

g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# 调整 FacetGrid
g.figure.subplots_adjust(hspace=-0.25)
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

# 显示图形

plt.show()



# 渐变填充函数gradient_fill_density分析：
# (1)kde = sns.kdeplot(data["x"], bw_adjust=0.5, clip_on=False, fill=False, alpha=1, linewidth=1.5)
# data["x"]: 数据列 x，即核密度估计的输入数据。
# bw_adjust=0.5: 调整带宽，带宽越小，曲线越窄。
# clip_on=False: 禁用剪裁，确保曲线完整绘制。
# fill=False: 不填充曲线，仅绘制边界。
# alpha=1: 设置曲线不透明度为 1。
# linewidth=1.5: 曲线宽度设置为 1.5。
# (2) x_vals = kde.get_lines()[0].get_xdata() y_vals = kde.get_lines()[0].get_ydata()
# 从 kde 对象中提取曲线数据。
# get_lines()[0]：获取绘制的第一条曲线（因为每次只绘制一条）。
# get_xdata() 和 get_ydata()：分别获取曲线的横坐标和纵坐标数据。
#
# (3)创建渐变色背景
# np.linspace(0, 1, len(x_vals))：生成从 0 到 1 的渐变数据，与 x_vals 的长度一致。
# .reshape(1, -1)：将数据调整为二维数组（1 行，多列）。
#
# cmap = parula：指定渐变色映射为 parula。
#
# extent=[x_vals.min(), x_vals.max(), 0, y_vals.max()]：
# 横向范围覆盖 x_vals 的最小值到最大值。
# 纵向范围覆盖从 0 到 y_vals 的最大值。
#
# ax.imshow 在子图中绘制渐变背景：
# aspect="auto"：背景自适应形状比例。
# extent 决定背景的位置和范围。
# im 是绘制的背景图像对象。
#
# fill_line, = ax.fill(x_vals, y_vals, facecolor="none", edgecolor="none")
# 创建曲线填充区域，但不直接填充颜色（facecolor="none"）。
# 返回填充区域的路径对象 fill_line。
#
# ax.plot(x_vals, y_vals, color="k", lw=1.5)  # 绘制边界线 绘制密度曲线边界线（黑色，线宽 1.5）
#
# im.set_clip_path(fill_line)：将背景图像裁剪到密度曲线的填充范围内。
#
#