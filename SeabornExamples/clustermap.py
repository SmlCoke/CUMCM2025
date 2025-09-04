import pandas as pd
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

# Load the brain networks example dataset
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Select a subset of the networks
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# Create a categorical palette to identify the networks
network_pal = sns.husl_palette(8, s=.45)
network_lut = dict(zip(map(str, used_networks), network_pal))

# Convert the palette to vectors that will be drawn on the side of the matrix
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=df.columns).map(network_lut)

# Draw the full plot
g = sns.clustermap(df.corr(), center=0, cmap="vlag",
                   row_colors=network_colors, col_colors=network_colors,
                   dendrogram_ratio=(.1, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))
# center=0：颜色渐变以 0 为中心。
# cmap="vlag"：使用 vlag 配色方案（红蓝对比）。
# row_colors=network_colors 和 col_colors=network_colors：在热图边缘绘制颜色条，表示网络归属。
# dendrogram_ratio=(.1, .2)：设置行/列的树状图比例。
# cbar_pos=(.02, .32, .03, .2)：设置颜色条的位置和大小。
# linewidths=.75：热图格子线宽。
# figsize=(12, 13)：设置图形尺寸。

g.ax_row_dendrogram.remove()
# 删除行的树状图，保留列的树状图。
plt.show()