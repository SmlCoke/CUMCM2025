from string import ascii_letters
import numpy as np
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

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# np.ones_like(corr, dtype=bool)：生成与相关性矩阵相同大小的布尔矩阵，全为 True。
# np.triu(...)：取矩阵的上三角部分，其他部分为 False。
# 作用: 通过掩膜隐藏上三角的值，只显示相关矩阵的下三角。
print(corr)
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('对角热力图.pdf')

# 热力图参数
# corr: 指定输入的相关性矩阵。
# mask=mask: 应用掩膜，仅显示下三角部分。
# cmap=cmap: 使用自定义发散型调色盘。
# vmax=.3: 将色彩范围的最大值设为 0.3，确保突出低相关性。 (因为相关系数矩阵中系数绝对值最大值不超过0.3，如果设置上限为1，则颜色对比不明显)
# center=0: 设置颜色中点为 0，便于区分正负相关。
# square=True: 将每个单元格设置为正方形。
# linewidths=.5: 单元格之间的边界线宽度为 0.5。
# cbar_kws={"shrink": .5}: 控制颜色条大小，缩小为默认高度的 50%。