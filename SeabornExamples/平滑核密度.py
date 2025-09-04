import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(
    context="paper",  # 设置上下文为 "paper"，适合学术论文
    style="white",  # 使用白色网格背景
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

df = sns.load_dataset("penguins")

g = sns.JointGrid(data=df, x="body_mass_g", y="bill_depth_mm", space=0.2)
g.plot_joint(sns.kdeplot,
             fill=True, clip=((2200, 6800), (10, 25)),
             thresh=0, levels=100, cmap="rocket")
g.plot_marginals(sns.histplot, color="#7C2333", alpha=1, bins=25)
plt.savefig('JointGrid_2.pdf')
plt.show()





# sns.JointGrid()的这套方法相比sns.jointplot()，调控更为精细

# g.plot_joint()参数
# fill=True：填充密度区域，生成颜色渐变效果。
# clip=((2200, 6800), (10, 25))：限制横轴（体重）和纵轴（喙深）的值域。
# thresh=0：设置显示阈值，0 表示显示所有密度范围。
# levels=100：绘制 100 等高线级别，提供更精细的密度渐变效果。
# cmap="rocket"：指定颜色映射表（"rocket" 渐变）