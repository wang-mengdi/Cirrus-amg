import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv

# 定义隐式函数
def f(x, y, z):
    # sphere centered at (0.5,0.5,0.5) with radius 0.25
    return (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2 - 0.25**2

# 网格采样
N = 256
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
z = np.linspace(0, 1, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 计算体素值
values = f(X, Y, Z)

# Marching Cubes 提取等值面 f=0
verts, faces, normals, _ = marching_cubes(
    values, level=0.0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0])
)

# 将 faces 转换为 pyvista 格式（每个三角形前面加上点数3）
faces_pv = np.hstack([[3, *face] for face in faces])

# 创建 PolyData 对象
mesh = pv.PolyData(verts, faces_pv)

# 导出为 .vtu 文件
mesh.save("isosurface.vtp")
