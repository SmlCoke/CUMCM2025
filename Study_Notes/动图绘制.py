"""
Python绘制动态图(GIF)学习笔记
基于matplotlib.animation模块实现

核心步骤：
1. 创建初始画布和图形元素
2. 定义动画更新函数
3. 使用FuncAnimation创建动画对象
4. 保存为GIF文件
"""

# ------------------------- 一、模块导入 -------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # 动画模块核心类

from matplotlib.font_manager import FontProperties # 设置字体
S_16 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 16)  # 宋体
S_14 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 14)  # 宋体
S_12 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 12)  # 宋体
T_12 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 12)



# ------------------------- 二、数据准备 -------------------------
# 生成x轴数据（固定不变）
x = np.linspace(start=-100, stop=100, num=200)

# 生成动态参数序列（示例为指数递减序列）
a_values = np.logspace(0, -5, num=100)  # 从10^0到10^-5生成100个点

# ------------------------- 三、图形初始化 -------------------------
# 创建画布和子图（必须在动画函数外初始化）
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# 初始化静态图形（左图）
y_sgn = np.concatenate([np.full(100, -1), np.full(100, 1)])
axes[0].plot(x, y_sgn,color = 'black')
axes[0].set_title("符号函数",fontproperties=S_14)
axes[0].set_xlim(-100,100)                 # x方向显示范围
axes[0].set_ylim(-1.5,1.5)                 # y方向显示范围
axes[0].set_yticks([-1,0,1])               # y轴显示的刻度
axes[0].set_xlabel(r'$t$',fontproperties=T_12)  # x轴标签
axes[0].set_ylabel(r'$f(t)$',fontproperties=T_12)  # y轴标签




# 设置动态图的坐标范围（固定范围保证动画稳定）
line, = axes[1].plot([], [], lw=2,color = 'black')  # 创建空线条对象
axes[1].set_xlim(-100, 100)
axes[1].set_ylim(-1.5, 1.5)
axes[1].set_title("双边指数信号",fontproperties=S_14)
axes[1].set_yticks([-1,0,1])
axes[1].set_xlabel(r'$t$',fontproperties=T_12)
axes[1].set_ylabel(r'$f(t)$',fontproperties=T_12)
axes[1].legend()


# ------------------------- 四、动画函数定义 -------------------------
# 初始化动态元素（文本）
text_annotation = axes[1].text(0.05, 0.95, "", transform=axes[1].transAxes,
                               fontproperties = S_12, color="red", weight="bold")

def init():
    """初始化函数，清空动态元素内容"""
    line.set_data([], [])
    text_annotation.set_text("")
    return line, text_annotation  # 必须返回所有需要更新的对象


def update(a):

    # 动画更新函数（每帧调用）
    # ★★★★参数a会自动从frames参数中取值


    # 计算新的y值
    y_e1 = -np.exp(a * x[:100])
    y_e2 = np.exp(-a * x[100:])

    # 更新图形元素
    line.set_data(x, np.concatenate([y_e1, y_e2]))
    text_annotation.set_text(f"Current a: {a:.3f}")  # 格式化显示3位小数

    return line, text_annotation  # 返回需要更新的对象列表


# ------------------------- 五、创建动画对象 -------------------------
ani = FuncAnimation(
    fig=fig,  # 目标画布
    func=update,  # 更新函数
    frames=a_values,  # 参数序列（自动传递给update）
    init_func=init,  # 初始化函数
    blit=True,  # 使用blitting优化性能（必须返回所有动态元素）
    interval=70,  # 帧间隔时间（单位：毫秒）
    repeat=True  # 是否循环播放
)

# ------------------------- 六、保存为GIF -------------------------
# 需要先安装pillow库：pip install pillow
ani.save("animation.gif", writer="pillow", dpi=100)  # dpi控制分辨率

# 显示图形（在Jupyter中可能需要使用%matplotlib notebook）
plt.show()

"""
学习要点总结：
1. 动画原理：
   - 通过连续调用update函数刷新图形元素
   - init函数用于初始化时清空内容
   - frames参数序列驱动动画变化

2. 性能优化：
   - 设置blit=True可显著提升性能（仅重绘变化部分）
   - 预先设置坐标轴范围（避免自动调整导致的闪烁）

3. 动态元素控制：
   - 文本更新使用set_text()
   - 线条更新使用set_data()
   - 所有动态元素必须在update函数中返回（blit=True时）

4. 参数调整技巧：
   - interval控制播放速度（值越小越快）
   - np.logspace生成对数间隔参数更适用于指数变化
   - 使用transAxes坐标系可实现相对位置定位

常见问题处理：
1. 文本不更新：检查是否将文本对象添加到返回列表中
2. 画面闪烁：确保设置了固定的坐标轴范围
3. 保存失败：确认已安装pillow库
"""