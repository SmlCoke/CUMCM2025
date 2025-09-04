import seaborn as sns

import matplotlib.pyplot as plt
sns.set_theme(
    context="paper",  # 设置上下文为 "paper"，适合学术论文
    style="dark",  # 使用白色网格背景
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

flights = sns.load_dataset("flights")

# Plot each year's time series in its own facet
g = sns.relplot(
    data=flights,
    x="month", y="passengers", col="year", hue="year",
    kind="line", palette="crest", linewidth=4, zorder=5,
    col_wrap=3, height=2, aspect=1.5, legend=False,
)
# col_wrap = 3 每行最多3列

# Iterate over each subplot to customize further
for year, ax in g.axes_dict.items():

    # Add the title as an annotation within the plot
    ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

    # Plot every year's time series in the background
    sns.lineplot(
        data=flights, x="month", y="passengers", units="year",
        estimator=None, color=".7", linewidth=1, ax=ax,
    )
# ax = ax控制将图绘制在哪个子图

# Reduce the frequency of the x axis ticks
ax.set_xticks(ax.get_xticks()[::2])             # 调成x轴刻度，防止过于密集

# Tweak the supporting aspects of the plot
g.set_titles("")
g.set_axis_labels("", "Passengers")
g.tight_layout()
plt.savefig('多时间序列.pdf')
plt.show()


# transform = ax.transAxes:
# transform=ax.transAxes：坐标系的变换方式。
# 默认情况下，ax.text 使用数据坐标（即以 x 和 y 的数据值为单位）。
# 指定 transform=ax.transAxes 后，坐标系统变为相对于轴的归一化坐标 [0, 1]。
# 这意味着无论数据范围如何变化，文本都保持在轴的相对位置。

# units="year"：
# 用于指定分组变量（year）的独立时间序列。
# 当指定此参数时，绘制的每条线代表一个唯一的 year，而不会对相同年份的 x 值（month）的数据进行聚合。
# 通常和 estimator=None 一起使用。

# estimator=None
# 指定是否对数据进行聚合。
# 默认情况下，sns.lineplot 会对相同的 x 值分组后计算统计量（如均值），绘制聚合后的折线。
# 设置为 None 后，取消聚合，每个分组的数据点直接绘制为单独的线。