import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('text', usetex=True)
sns.set_theme(
    context="paper",  # 设置上下文为 "paper"，适合学术论文
    style="whitegrid",  # 使用白色网格背景
    font_scale=1.2,  # 全局字体缩放比例
    rc={
        "font.family": "serif",  # 使用衬线字体（如 Times New Roman）
        "font.serif": ["Times New Roman"],  # 指定衬线字体
        "axes.labelsize": 15,  # 轴标签字体大小
        "xtick.labelsize": 15,  # x轴刻度字体大小
        "ytick.labelsize": 15,  # y轴刻度字体大小
        "legend.fontsize": 10,  # 图例字体大小
        "axes.titlesize": 16,  # 图标题字体大小
        "figure.titlesize": 18,  # 整体图的标题字体大小
        "grid.linewidth": 0.5,  # 网格线宽度
        "lines.linewidth": 1.5,  # 线条宽度
        "axes.linewidth": 1.2,  # 坐标轴线宽度
        "patch.linewidth": 1.2,  # 图形边框宽度
    }
)


df = pd.read_excel('Num=0.8.xlsx')


fig = plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid(shape =(2,3),loc = (0,0))
ax2 = plt.subplot2grid(shape = (2,3), loc = (1,0))
ax3 = plt.subplot2grid(shape = (2,3), loc = (0,1), rowspan = 2,colspan=2)



bins1 = [0.5,0.53,0.65,0.85,1.5,1.8,2.2]
labels1 = ['$= 0.5$',r'$\approx 0.55$',r'$\approx 0.71$',r'$\approx 1.0$',r'$\approx 1.66$','$= 2$']


df['Range'] = pd.cut(df['Env_co'], bins=bins1, labels=labels1, right=False)
distribution = df['Range'].value_counts().sort_index()


# 在 ax1 中绘制饼图
ax1.pie(
    distribution,
    labels=distribution.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.tab20.colors
)
ax1.set_title(r'Distribution of $E_c$')



# 定义典型值
typical_values = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])

# 计算每个数据点到典型值的距离，找到最近的典型值
df['Closest'] = df['Tax_up'].apply(lambda x: typical_values[np.abs(typical_values - x).argmin()])

# 统计每个典型值的出现次数
distribution = df['Closest'].value_counts().sort_index()

ax2.pie(
    distribution,
    labels=distribution.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.tab20.colors
)
ax2.set_title(r'Distribution of $T_u$')

sns.kdeplot(
    data=df,
    x='Env_co',
    y='Tax_up',
    cmap='viridis',
    fill=True,
    ax=ax3             # ★★★★ 将该图送如子图ax3，这个操作实现了mat图与sea图同时绘制在一个窗口
)
ax3.set_title(r'KDE of $T_u$ and $E_c$')
ax3.set_xlabel(r'$E_c$')
ax3.set_ylabel(r'$T_u$')

# 显示图形
plt.tight_layout()
plt.savefig('第二问决策分布图.pdf')
plt.show()