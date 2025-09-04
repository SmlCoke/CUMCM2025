import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="dark")
pal = sns.color_palette("Paired", n_colors=2)
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

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker",
               split=True, inner="quart", fill=True,
               palette=pal)
plt.savefig('分割小提琴.pdf')
plt.show()

# tips:
# inner
# 作用：控制小提琴图内部显示的内容。
# 值：
# "quart"：显示四分位数（即数据分布的 25%、50% 和 75% 分位）。
# "box"：显示类似箱线图的统计信息。
# "point"：在小提琴图中显示数据点。
# "stick"：用竖线表示数据点。
# None：不显示内部内容。
# 此处为 "quart"，显示分布的四分位数。