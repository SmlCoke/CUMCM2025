import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(
    context="paper",  # 设置上下文为 "paper"，适合学术论文
    style="ticks",  #
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
hls_palette = sns.color_palette("hls", n_colors=10)     # hls调色板

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(10, 7))

ax.set_xscale("log")
# 设置 x 轴为对数刻度 (log)，适合展示具有较大范围差异的变量（如天文数据中的距离）。

# Load the example planets dataset
planets = sns.load_dataset("planets")

# Plot the orbital period with horizontal boxes
sns.boxplot(
    planets, x="distance", y="method", hue="method",
    whis=[0, 100], width=.6, palette=hls_palette
)

# Add in points to show each observation
sns.stripplot(planets, x="distance", y="method", size=4, color=".3")

# Tweak the visual presentation
plt.yticks(rotation=45)
ax.set(ylabel="")
sns.despine(trim=True)
plt.savefig('含观测值的水平箱线图.png')
plt.show()