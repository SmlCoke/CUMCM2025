import matplotlib.pyplot as plt
import seaborn as sns
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

# Load the example flights dataset and convert to long-form
flights_long = sns.load_dataset("flights")
flights = (
    flights_long
    .pivot(index="month", columns="year", values="passengers")
)

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.savefig('带注释的热图.pdf')



# 注：
# annot = True: 在热力图的每个单元格中显示数据值。默认值为False
# fmt="d": 设置显示数据值的格式:
#   "d"：整数格式，例如 12、45。
#   ".2f"：保留两位小数的浮点数格式，例如 12.34。
#   ".1f"：保留一位小数的浮点数格式，例如 5.6。
#   "e"：科学计数法，例如 1.23e+03。
# ax=ax: 将热力图绘制到指定的 Matplotlib Axes 对象上。 默认值：none: 自动创建一个新的绘图区域 见脚本多子图热力图