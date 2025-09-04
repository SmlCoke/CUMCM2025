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
hls_palette = sns.color_palette("hls", n_colors=3)     # hls调色板
df = sns.load_dataset("penguins")
sns.pairplot(df, kind = 'scatter', diag_kind= 'kde', hue="species",palette= hls_palette)
plt.show()

# pairplot用于绘制成对变量关系，能够同时显示dataframe中的所有数值变量之间的关系
# pairplot参数
# data：必选参数，输入的数据集（通常是 Pandas 的 DataFrame）。
# hue：指定分类变量，区分不同类别的点（例如：性别、地区等），并为其分配不同颜色。
# kind：设置非对角线图的类型：
#   "scatter"：散点图（默认值）。
#   "kde"：核密度估计图。
#  diag_kind：设置对角线图的类型：
#   "auto"：根据 kind 自动选择。
#   "hist"：直方图。
#   "kde"：核密度估计图。
# palette：设置不同类别的颜色映射方案。
# markers：为不同类别的点设置标记符号。
# corner：如果为 True，则只绘制变量之间的下三角部分，而不绘制上三角部分。


# 如果只想要研究部分变量的全关系：
# sns.pairplot(iris, vars=["sepal_length", "sepal_width", "petal_length"])
# 如果想要设置 x 和 y 轴分别对应的变量（即对变量组指定轴），可以使用 x_vars 和 y_vars：
# sns.pairplot(iris, x_vars=["sepal_length", "sepal_width"], y_vars=["petal_length", "petal_width"])