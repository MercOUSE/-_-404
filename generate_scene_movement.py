import numpy as np
import open3d as o3d

# Размерность облака точек
num_points = 1000
a = 20  # Размер куба от -a до a по каждой оси (например, куб от -1 до 1)

# Разбиваем пространство на сетку для равномерного распределения точек
grid_size = int(np.cbrt(num_points))  # Определяем размер сетки, примерно равный кубическому корню от числа точек
x = np.linspace(-a, a, grid_size)
y = np.linspace(-a, a, grid_size)
z = np.linspace(-a, a, grid_size)

# Генерация равномерного распределения точек внутри куба
X, Y, Z = np.meshgrid(x, y, z)

# Преобразуем сетку в список точек
points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# Убираем точки, которые находятся внутри куба (оставляем только те, которые на границе)
# Точки, находящиеся на границе, имеют хотя бы одно значение x, y, или z равным ±a
border_points = points[np.any(np.abs(points) == a, axis=1)]

# Создание объекта облака точек
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(border_points)

# Сохранение в .ply файл
o3d.io.write_point_cloud("point_cloud.ply", point_cloud)

# Выводим на экран облако точек
o3d.visualization.draw_geometries([point_cloud])