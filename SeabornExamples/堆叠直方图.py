import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
K_16 = FontProperties(fname = r'C:\\Windows\\Fonts\\STKAITI.TTF', size = 16) # 楷体
sns.set_theme(
    context="paper",  # 设置上下文为 "paper"，适合学术论文
    style="whitegrid",  # 使用白色网格背景
    font_scale=1.2,  # 全局字体缩放比例
    rc={
        "font.family": "serif",  # 使用衬线字体（如 Times New Roman）
        "font.serif": ["Times New Roman"],  # 指定衬线字体
        "axes.labelsize": 14,  # 轴标签字体大小
        "xtick.labelsize": 12,  # x轴刻度字体大小
        "ytick.labelsize": 12,  # y轴刻度字体大小
        "legend.fontsize": 10,  # 图例字体大小
        "axes.titlesize": 14,  # 图标题字体大小
        "figure.titlesize": 16,  # 整体图的标题字体大小
        "grid.linewidth": 1,  # 网格线宽度
        "lines.linewidth": 1.5,  # 线条宽度
        "axes.linewidth": 1.2,  # 坐标轴线宽度
        "patch.linewidth": 0.8,  # 图形边框宽度
    }
)

# 加载数据集
tips = sns.load_dataset("tips")



# 绘制堆叠直方图
sns.histplot(
    data=tips,
    x="total_bill",    # x轴变量：账单总额
    hue="sex",         # 分类变量：性别
    multiple="stack",  # 设置为堆叠模式
    palette="coolwarm" # 调色板
)

# 添加标题和标签
plt.title("堆叠直方图案例", fontproperties = K_16)
plt.xlabel("Total Bill ($)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.savefig('堆叠直方图.pdf')
plt.show()
