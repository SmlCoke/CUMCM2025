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

# Load the brain networks dataset, select subset, and collapse the multi-index
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns
                  .get_level_values("network")
                  .astype(int)
                  .isin(used_networks))
df = df.loc[:, used_columns]

df.columns = df.columns.map("-".join)


# Compute a correlation matrix and convert to long-form
corr_mat = df.corr().stack().reset_index(name="correlation")
# .corr(): 计算相关矩阵
# .stack(): 将宽格式的矩阵 "压缩" 为长格式，列名作为一个新维度。
# 双重列索引，n×n，第三列为correlation

# Draw each cell as a scatter point with varying size and color
g = sns.relplot(
    data=corr_mat,
    x="level_0", y="level_1", hue="correlation", size="correlation",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(50, 250), size_norm=(-.2, .8),
)

# Tweak the figure to finalize
g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)
plt.savefig('散点热图.pdf')
plt.show()


# 相关矩阵通常通过热图进行可视化
# 参数说明：
# hue="correlation"：用颜色表示相关系数大小，颜色由调色板 palette 控制。
# size="correlation"：用点的大小表示相关系数大小。
# palette="vlag"：使用 Seaborn 提供的配色方案 "vlag"，适合表示正负值的渐变。
# hue_norm=(-1, 1)：将相关系数的颜色范围标准化到 [-1, 1]。
# edgecolor=".7"：点的边框颜色为灰色，增强可视化效果。
# height=10：图表高度（单位：英寸）。
# sizes=(50, 250)：点大小范围（最小值为 50，最大值为 250）。
# size_norm=(-.2, .8)：将相关系数的大小范围标准化到 [-0.2, 0.8]。

# 图表设置
# g.set(xlabel="", ylabel="", aspect="equal")：去掉 x 轴和 y 轴的标签。设置图表的宽高比为 1：1，保持方形单元格。
# g.despine(left=True, bottom=True)：去掉左侧和底部的坐标轴框线。
# g.ax.margins(.02)：设置图表的边距为 2%。
# for label in g.ax.get_xticklabels():
#     label.set_rotation(90)
# 作用：将 x 轴的标签旋转 90 度，避免重叠。