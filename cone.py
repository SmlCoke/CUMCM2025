# Python code to plot the 3D scene (ellipse, point light P, plane x=d, projection curve)
# and a separate 2D plot of the projection curve (y vs z on the plane x=d).
# This uses matplotlib (mpl_toolkits.mplot3d) and numpy.
# Defaults follow your constraint: xp > d > a > b.
# You can modify parameters a,b,xp,yp,zp,d as needed and re-run.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- parameters (modify these) ----------
a = 2.0      # ellipse semi-axis along x
b = 1.0      # ellipse semi-axis along y
x_p = 4.0    # P.x (must satisfy x_p > d and x_p > a)
y_p = 0.0    # P.y
z_p = 3.0    # P.z  (height)
d   = 3.0    # plane x = d  (must satisfy d > a)
n_points = 800  # resolution for parameter t
# ------------------------------------------------

# quick sanity check (prints a warning if constraints violated)
if not (x_p > d > a > b):
    print("警告：参数不满足 xp > d > a > b，绘图仍会尝试进行，但结果可能包含无穷或失真。")

# param t and ellipse points
t = np.linspace(0, 2*np.pi, n_points)
x_e = a * np.cos(t)
y_e = b * np.sin(t)
z_e = np.zeros_like(t)

# compute lambda for intersection with plane x = d
den = x_e - x_p
eps = 1e-12
mask = np.abs(den) > eps  # safe points where denominator isn't (near) zero

lambda_vals = np.full_like(t, np.nan)
lambda_vals[mask] = (d - x_p) / den[mask]

# intersection points on plane
X_proj = np.full_like(t, d)
Y_proj = np.full_like(t, np.nan)
Z_proj = np.full_like(t, np.nan)
Y_proj[mask] = y_p + lambda_vals[mask] * (y_e[mask] - y_p)
Z_proj[mask] = z_p + lambda_vals[mask] * (0.0 - z_p)  # = z_p*(1 - lambda)

# ------- 3D plot: ellipse, P, plane, projection, and sample rays -------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# original ellipse (z=0)
ax.plot(x_e, y_e, z_e, linewidth=2, label='Ellipse (z=0)')

# projected curve on plane x=d
ax.plot(X_proj[mask], Y_proj[mask], Z_proj[mask], linestyle='--', linewidth=2, label='Projection on plane x=d')

# point light P
ax.scatter([x_p], [y_p], [z_p], s=40)
ax.text(x_p, y_p, z_p, "  P", verticalalignment='bottom')

# draw plane x=d as a small rectangular patch covering the projection extents
pad_y = 0.2 * (np.nanmax(Y_proj) - np.nanmin(Y_proj) + 1.0)
pad_z = 0.2 * (np.nanmax(Z_proj) - np.nanmin(Z_proj) + 1.0)
ymin, ymax = np.nanmin(Y_proj) - pad_y, np.nanmax(Y_proj) + pad_y
zmin, zmax = np.nanmin(Z_proj) - pad_z, np.nanmax(Z_proj) + pad_z
ys = np.linspace(ymin, ymax, 2)
zs = np.linspace(zmin, zmax, 2)
Y_mesh, Z_mesh = np.meshgrid(ys, zs)
X_mesh = np.full_like(Y_mesh, d)
# translucent surface to indicate the screen/plane
ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.25, rstride=1, cstride=1, linewidth=0, antialiased=True)

# draw a few sample rays (connect P -> ellipse point -> plane intersection)
num_rays = 24
idxs = np.linspace(0, n_points-1, num_rays, dtype=int)
for i in idxs:
    if mask[i]:
        seg_x = [x_p, x_e[i], d]
        seg_y = [y_p, y_e[i], Y_proj[i]]
        seg_z = [z_p, z_e[i], Z_proj[i]]
        ax.plot(seg_x, seg_y, seg_z, linewidth=0.8)

# axis labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# make axis aspect roughly equal
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

# adjust limits to include all features
all_x = np.concatenate(([x_p], x_e, X_proj[mask]))
all_y = np.concatenate(([y_p], y_e, Y_proj[mask]))
all_z = np.concatenate(([z_p], z_e, Z_proj[mask]))
ax.set_xlim(np.min(all_x)-0.5, np.max(all_x)+0.5)
ax.set_ylim(np.min(all_y)-0.5, np.max(all_y)+0.5)
ax.set_zlim(np.min(all_z)-0.5, np.max(all_z)+0.5)
set_axes_equal(ax)

ax.set_title('3D view: Ellipse (z=0), point P, plane x=d and projected curve')
ax.legend()
plt.show()

# ------- 2D plot: projection curve on plane coordinates (Y vs Z) -------
fig2 = plt.figure(figsize=(6,5))
ax2 = fig2.add_subplot(111)
ax2.plot(Y_proj[mask], Z_proj[mask], linewidth=2)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlabel('y (on plane x=d)')
ax2.set_ylabel('z (on plane x=d)')
ax2.set_title('Projection curve on plane x=d (y vs z)')
plt.grid(True)
plt.show()

# ------- optional: print a short numeric summary -------
print("Parameters: a=%.3g, b=%.3g, P=(%.3g, %.3g, %.3g), d=%.3g" % (a, b, x_p, y_p, z_p, d))
valid_fraction = np.count_nonzero(mask) / mask.size
print("可计算的参数点比例（排除了极点/并行情况）： %.3f" % valid_fraction)
