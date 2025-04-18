import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv

# 定义隐式函数
def f(x, y, z):
    # 示例：球面
    return x**2 + y**2 + z**2 - 1.0

# 网格采样
N = 100
x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1.5, 1.5, N)
z = np.linspace(-1.5, 1.5, N)
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
mesh.save("isosurface.vtu")
