# 导入模块
'''
import matplotlib.pyplot as plt
'''

# 图形正确显示
'''
plt.rc('font', family="SimHei")            # 正确显示中文
plt.rc('font', size=16)                    # 设置字号
plt.rc('axes',unicode_minus = False)       # 正确显示负号
plt.rcParams['text.usetex'] = True         # 调用LaTeX方法渲染文本
'''

# 打开一个新的图形窗口
''' 
plt.figure(figsize = (10,8))
或fig = plt.figure()获取窗口名
'''

# 子图的绘制
'''
方式一:
plt.figure()
plt.subplot(abc) : a行b列第c个(c是索引，从1开始)
使用此命令之后，plt自动移至当前子图窗口
获取轴：ax = plt.gca()

方式二:
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(222)

# 调整布局
plt.tight_layout()

方式三:一次创建一组子图
fig,axes = plt.subplots(a,b)
创建a行b列的图形窗口
axes看成矩阵，具有切片操作
'''

# ax与plt
'''
ax = plt.gca() 用于获取当前的 Axes 对象。
Axes 对象是 matplotlib 中用于绘制图形的核心对象，
它包含了图形的所有元素，如线条、标记、标签、标题等。
ax = plt.gca()
'''

# 轴
'''
plt.xticks([num1, num2, num3,...,num_n])  将x轴的刻度显示为数值num1, num2,...
同时x轴范围（或者说整个子图x方向的范围）变为num1~num_n
plt.xticks([num1, num2, num3,...,num_n], [str1, str2,... str_n])
显示为刻度大小对应于数值num1, num2,...的刻度
同时x轴范围（或者说整个子图x方向的范围）变为num1~num_n
并且将刻度的符号从num1, num2, ... 修改为字符串str1, str2,...str_n
想要显示的字符串同样支持r'$...$'方法

'''

# 三维曲面
'''
除了基础模块matplotlib.pyplot外还需导入：
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (10,8))
x = list
x,y = np.meshgrid(x,x)
ax = fig.add_subplot(111,projection = '3d')
z = func(x,y)
ax.plot_surface(x,y,z,cmap = "viridis" , color = "green",label = "z = func(x,y)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('目标函数')
plt.legend(loc = "upper left" )
'''

# 三维曲线
'''
除了matplotlib.pyplot外不需要额外导入库
plt.subplot(111,projection= '3d')
给出x,y,z的参数方程及参数样本点集(np.linspace(a,b,num))
plt.plot(x,y,z,color = 'green', label = "test")
plt.legend(loc = "upper left" , title = "Test")
'''

# 设置字体
'''
from matplotlib.font_manager import FontProperties
S_16 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 16)  # 宋体
K_16 = FontProperties(fname = r'C:\\Windows\\Fonts\\STKAITI.TTF', size = 16) # 楷体
T_16 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 16)    # Times New Roman
S_14 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 14)  # 宋体
K_14 = FontProperties(fname = r'C:\\Windows\\Fonts\\STKAITI.TTF', size = 14) # 楷体
T_14 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 14) 
S_12 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 12)  # 宋体
K_12 = FontProperties(fname = r'C:\\Windows\\Fonts\\STKAITI.TTF', size = 12) # 楷体
T_12 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 12)
r 前缀表示原始字符串，它告诉Python忽略字符串中的转义字符，如反斜杠 \
设置方式:
plt.figure()型：
plt.xlabel(fontproperties = font1)
plt.title(fontproperties = font1)
plt.legend(prop = font1)

fig,ax = plt.subplots()型：
ax.set_xlabel(fontproperties = font2)
ax.set_title(fontproperties = font1)
fig.legend(prop = font1
'''

# 将刻度转化为百分数
'''
def to_percentage(x, pos):
    return f'{100 * x:.2f}%'

fig,ax = plt.subplots()型：
ax.xaxis.set_major_formatter(FuncFormatter(to_percentage))

plt.figure()型：
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percentage))
'''