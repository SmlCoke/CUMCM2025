import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset('tips')
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
g=sns.catplot(x="sex", y="total_bill", hue="smoker",data=tips, kind="violin", col = 'time')

# 更改轴的标签及字体字号
for ax in g.axes.flat:  # 遍历所有子图的轴
    ax.set_xlabel("Custom X Label", fontdict={"fontsize": 12, "fontweight": "bold", "family": "serif"})
    ax.set_ylabel("Custom Y Label", fontdict={"fontsize": 12, "fontweight": "bold", "family": "serif"})
plt.show()