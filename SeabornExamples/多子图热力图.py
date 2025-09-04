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
flights_long = sns.load_dataset("flights")
flights = (
    flights_long
    .pivot(index="month", columns="year", values="passengers")
)

# 创建 1 行 2 列的子图布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 在第一个子图中绘制热力图
sns.heatmap(flights, annot=True, fmt="d", ax=ax1)
ax1.set_title("Subplot 1")

# 在第二个子图中绘制热力图
sns.heatmap(flights, annot=False, cmap="coolwarm", ax=ax2)
ax2.set_title("Subplot 2")
plt.savefig('多子图热力图.pdf')
plt.show()
