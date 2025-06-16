import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

def distance_and_color(pcd):
    """Пересчитывает расстояния до центра (0, 0, 0) и обновляет цвета."""
    points = np.asarray(pcd.points)
    lidar_position = np.array([0, 0, 0])
    
    # Вычисляем расстояние до лидара
    distances = np.linalg.norm(points - lidar_position, axis=1)
    
    # Нормализация расстояний (от 0 до 1)
    distances_normalized = (distances - distances.min()) / (distances.max() - distances.min())

    # Применяем цветовую карту
    cmap = plt.get_cmap("RdBu")  
    colors = cmap(distances_normalized)[:, :3]  # Оставляем только RGB

    pcd.colors = o3d.utility.Vector3dVector(colors)

def load_point_cloud(file_path):
    """Загружает облако точек и задает его начальную позицию (0, 0, 70)."""
    pcd = o3d.io.read_point_cloud(file_path)

    # Смещение в начальную позицию (0,0,70)
    initial_position = np.array([0, 0, 70])
    points = np.asarray(pcd.points)
    moved_points = points + (initial_position - np.mean(points, axis=0))
    pcd.points = o3d.utility.Vector3dVector(moved_points)

    # Сфера в центре (0,0,0) для ориентира
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    sphere.paint_uniform_color([1, 1, 0])  # Желтый цвет сферы
    sphere.translate([0, 0, 0])  # Размещаем сферу в центре

    distance_and_color(pcd)  # Применяем цвета
    return pcd, sphere, moved_points

def animate_point_cloud(file_path, duration=20, frequency=1, amplitude=120):
    """Анимация движения куба влево-вправо относительно (0, 0, 70)."""
    pcd, sphere, original_points = load_point_cloud(file_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(sphere)

    start_time = time.time()

    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        shift = np.sin(elapsed_time * frequency) * amplitude  # Движение по X

        # Обновляем положение точек (X меняется, Z = 70)
        moved_points = np.copy(original_points)
        moved_points[:, 0] += shift  # Двигаем только по оси X
        pcd.points = o3d.utility.Vector3dVector(moved_points)

        # Пересчет расстояний и цветов
        distance_and_color(pcd)

        # Обновляем данные в визуализаторе
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
    
    vis.destroy_window()

if __name__ == "__main__":
    file_path = "point_cloud.ply"  
    animate_point_cloud(file_path)
