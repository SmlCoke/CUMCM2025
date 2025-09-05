import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, cos, sin, simplify, expand, collect
from matplotlib.font_manager import FontProperties
S_14 = FontProperties(fname = r'C:\\Windows\\Fonts\\simsun.ttc', size = 14)  # 宋体

# =======================
# (1) 符号推导：隐式方程
# =======================

# 定义符号
a, b, xp, yp, zp, d, t = symbols('a b xp yp zp d t', real=True)

# 椭圆参数方程
xe = a*cos(t)
ye = b*sin(t)

# 光源 P
# P = (xp, yp, zp)

# 直线参数方程
lam = (d - xp) / (xe - xp)
y_proj = yp + lam*(ye - yp)
z_proj = zp + lam*(0 - zp)

# 消去 t 得到 y,z 关系
# 思路：将 cos(t), sin(t) 表达式用 y_proj, z_proj 替换，再整理
# 实际上我们要消元 t，最直接的方法是写一个等式，强制 y_proj,z_proj 必须满足椭圆的关系
#   (xe/a)^2 + (ye/b)^2 = 1

# 我们知道 xe = a*cos(t), ye = b*sin(t)，所以
#   cos(t) = xe/a, sin(t) = ye/b
# 椭圆条件: (xe/a)^2 + (ye/b)^2 = 1

eq = Eq( ( ( (y_proj - yp)*(xe - xp)/(d - xp) + yp )/b )**2 +
         ( ( ( (d - xp)*(0 - zp)/(xe - xp) + zp )/a )**2 ), 1)

# 不过这个形式比较复杂，我们直接代入投影公式再简化更稳妥
# 这里更直接：投影曲线满足一个二次方程
Y, Z = symbols('Y Z', real=True)
lam_y = (Y - yp)/(ye - yp)
lam_z = (Z - zp)/(-zp)
# 消元条件: lam_y = lam_z = lam
eq_final = simplify(expand(lam_y - lam_z))

print("======= 投影曲线隐式方程（x=d 平面上） =======")
print("eq(y,z) = 0 的形式：")
print(eq_final)

# =======================
# (2) 数值绘制：3D & 2D
# =======================

from mpl_toolkits.mplot3d import Axes3D

# 参数设置
a_val, b_val = 2.0, 1.0
xp_val, yp_val, zp_val = 5.0, 0.5, 3.0
d_val = 4.0

# 椭圆参数
n_points = 800
t_vals = np.linspace(0, 2*np.pi, n_points)
x_e = a_val*np.cos(t_vals)
y_e = b_val*np.sin(t_vals)
z_e = np.zeros_like(t_vals)

# 投影
den = (x_e - xp_val)
mask = np.abs(den) > 1e-12
lam_vals = (d_val - xp_val) / den[mask]
X_proj = np.full_like(t_vals, d_val)
Y_proj = yp_val + lam_vals*(y_e[mask] - yp_val)
Z_proj = zp_val + lam_vals*(0 - zp_val)

# ---- 3D 绘制 ----
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

# 椭圆
ax.plot(x_e, y_e, z_e, label='Ellipse (z=0)', linewidth=2)

# 投影曲线
ax.plot(X_proj, Y_proj, Z_proj, '--', label='Projection (x=d)', linewidth=2)

# 光源 P
ax.scatter([xp_val], [yp_val], [zp_val], s=40, c='r')
ax.text(xp_val, yp_val, zp_val, "  P", verticalalignment='bottom')

# 幕布平面
ymin, ymax = np.min(Y_proj)-1, np.max(Y_proj)+1
zmin, zmax = np.min(Z_proj)-1, np.max(Z_proj)+1
ys = np.linspace(ymin, ymax, 2)
zs = np.linspace(zmin, zmax, 2)
Y_mesh, Z_mesh = np.meshgrid(ys, zs)
X_mesh = np.full_like(Y_mesh, d_val)
ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('3D 投影图: 椭圆、光源、幕布及投影曲线',fontproperties = S_14)

plt.show()

# ---- 2D 绘制 (y-z) ----
fig2, ax2 = plt.subplots(figsize=(6,5))
ax2.plot(Y_proj, Z_proj, linewidth=2)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlabel('y (plane x=d)')
ax2.set_ylabel('z (plane x=d)')
ax2.set_title('投影曲线在幕布上的形状 (y-z)',fontproperties = S_14)
plt.grid(True)
plt.show()
