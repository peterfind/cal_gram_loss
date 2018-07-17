# coding=UTF-8
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # 不跳出图像窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

txt_path = './outputs2/'


loss_save = np.loadtxt(os.path.join(txt_path, 'loss_save.txt'))

plt.plot(loss_save.reshape((-1)))  # 第一行、第二行 ***
plt.savefig(os.path.join(txt_path, 'loss_curve.png'))

fig = plt.figure()
ax = Axes3D(fig)
x = range(loss_save.shape[1])
y = range(loss_save.shape[0])
X, Y = np.meshgrid(x, y)
Z = loss_save
ax.plot_surface(X, Y, Z, cmap=plt.cm.winter)
plt.savefig(os.path.join(txt_path, 'loss_3D.png'))

