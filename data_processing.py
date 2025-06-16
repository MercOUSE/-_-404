import cv2
import numpy as np
import math
from collections import deque
import time
import random
from typing import List, Tuple, Dict, Optional

class DroneSimulator3D:
    def __init__(self):
        # Параметры 3D симуляции
        self.world_size = np.array([1000, 1000, 1000])  # Ширина, высота, глубина (в метрах)
        
        # Позиция цели и препятствия
        self.target_pos = np.array([900.0, 500.0, 500.0], dtype=np.float32)
        self.obstacle_pos = np.array([500.0, 500.0, 500.0], dtype=np.float32)
        self.obstacle_radius = 80
        
        # Случайная позиция дрона
        self.drone_pos = self.generate_random_start_position()
        
        # Параметры дрона
        self.drone_speed = 2.0
        self.trajectory = deque(maxlen=1000)
        self.running = True
        self.avoidance_direction = ""  # Текущее направление облета
        
        # Состояние навигации
        self.navigation_state = "approaching"
        self.avoid_start_pos = None
        self.avoid_start_time = 0
        
        # Параметры навигации
        self.avoid_distance = 150
        self.safety_margin = 100
        self.normal_heading = np.array([1.0, 0.0, 0.0])
        self.normal_heading = self.normal_heading / np.linalg.norm(self.normal_heading)
        
        # Параметры сенсоров
        self.lidar_range = 350
        self.lidar_noise_std = 1.5
        self.camera_fov = math.radians(70)
        self.camera_resolution = (640, 480)
        self.min_object_pixels = 80
        
        # Имитация потоковых данных
        self.last_sensor_update = 0
        self.sensor_data = {
            'lidar_points': [],
            'camera_area': 0,
            'closest_distance': float('inf'),
            'obstacle_detected': False
        }
        
        # Создание 2D дисплея
        self.display_size = (800, 600)
        self.display_img = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
        cv2.namedWindow('Drone 3D Navigation', cv2.WINDOW_NORMAL)

    def generate_random_start_position(self) -> np.ndarray:
        """Генерация случайной стартовой позиции"""
        while True:
            x = random.uniform(50, 400)
            
            if random.choice([True, False]):
                y = random.choice([
                    random.uniform(self.obstacle_pos[1] + self.obstacle_radius + 50, 700),
                    random.uniform(300, self.obstacle_pos[1] - self.obstacle_radius - 50)
                ])
                z = random.uniform(300, 700)
            else:
                y = random.uniform(300, 700)
                z = random.choice([
                    random.uniform(self.obstacle_pos[2] + self.obstacle_radius + 50, 700),
                    random.uniform(300, self.obstacle_pos[2] - self.obstacle_radius - 50)
                ])
            
            pos = np.array([x, y, z], dtype=np.float32)
            
            if self.is_path_blocked(pos, self.target_pos):
                return pos

    def is_path_blocked(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Проверка пересечения линии с препятствием"""
        direction = end - start
        distance = np.linalg.norm(direction)
        if distance == 0:
            return False
        
        direction = direction / distance
        obstacle_to_start = start - self.obstacle_pos
        cross_prod = np.linalg.norm(np.cross(obstacle_to_start, direction))
        min_distance = cross_prod / np.linalg.norm(direction)
        
        return min_distance < self.obstacle_radius

    def safe_acos(self, x: float) -> float:
        return math.acos(max(-1.0, min(1.0, x)))

    def generate_lidar_points(self) -> List[Tuple[np.ndarray, float]]:
        points = []
        obstacle_vec = self.obstacle_pos - self.drone_pos
        distance = np.linalg.norm(obstacle_vec)
        
        if distance > self.lidar_range:
            return points
        
        num_points = int(300 * (1 - distance/self.lidar_range))
        num_points = max(30, min(num_points, 300))
        
        for _ in range(num_points):
            theta = random.uniform(0, 2*math.pi)
            phi = random.uniform(0, math.pi)
            r = self.obstacle_radius * random.uniform(0.85, 1.15)
            
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            
            point = self.obstacle_pos + np.array([x, y, z]) + np.random.normal(0, self.lidar_noise_std, 3)
            points.append(point)
        
        return points

    def calculate_camera_area(self) -> int:
        obstacle_vec = self.obstacle_pos - self.drone_pos
        distance = np.linalg.norm(obstacle_vec)
        
        if distance == 0:
            return 0
        
        direction = obstacle_vec / distance
        dot_product = np.dot(direction, self.normal_heading)
        angle_to_obstacle = self.safe_acos(dot_product)
        
        if angle_to_obstacle > self.camera_fov/2:
            return 0
        
        apparent_radius = self.obstacle_radius / distance
        pixel_radius = apparent_radius * self.camera_resolution[0] / (2 * math.tan(self.camera_fov/2))
        area = math.pi * (pixel_radius ** 2) * random.uniform(0.7, 1.3)
        
        return min(int(area), self.camera_resolution[0]*self.camera_resolution[1])

    def get_sensor_data(self) -> Dict:
        current_time = time.time()
        if current_time - self.last_sensor_update < 0.2:
            return self.sensor_data
        
        lidar_points = self.generate_lidar_points()
        closest_distance = float('inf')
        
        distances = []
        for point in lidar_points:
            dist = np.linalg.norm(point - self.drone_pos)
            distances.append(dist)
            if dist < closest_distance:
                closest_distance = dist
        
        camera_area = self.calculate_camera_area()
        
        self.sensor_data = {
            'timestamp': current_time,
            'lidar_points': list(zip(lidar_points, distances)),
            'camera_area': camera_area,
            'closest_distance': closest_distance if lidar_points else float('inf'),
            'obstacle_detected': (closest_distance <= self.avoid_distance + self.obstacle_radius or 
                                camera_area >= self.min_object_pixels)
        }
        
        self.last_sensor_update = current_time
        return self.sensor_data

    def has_passed_obstacle(self) -> bool:
        obstacle_to_drone = self.drone_pos - self.obstacle_pos
        obstacle_to_target = self.target_pos - self.obstacle_pos
        
        dot_product = np.dot(obstacle_to_drone, obstacle_to_target)
        norm_product = np.linalg.norm(obstacle_to_drone) * np.linalg.norm(obstacle_to_target)
        
        if norm_product == 0:
            return False
            
        angle = math.acos(min(max(dot_product / norm_product, -1.0), 1.0))
        
        return (angle < math.pi/2) and (np.linalg.norm(obstacle_to_drone) > self.obstacle_radius + self.safety_margin)

    def calculate_avoidance_direction(self) -> Tuple[np.ndarray, str]:
        """Вычисление направления облета с правильными ориентациями"""
        to_obstacle = self.obstacle_pos - self.drone_pos
        distance = np.linalg.norm(to_obstacle)
        if distance == 0:
            return np.array([0.0, 0.0, 1.0]), "front"
        
        to_obstacle_normalized = to_obstacle / distance
        to_target = self.target_pos - self.drone_pos
        to_target_norm = np.linalg.norm(to_target)
        
        if to_target_norm == 0:
            return np.array([0.0, 0.0, 1.0]), "front"
        
        to_target_normalized = to_target / to_target_norm
        
        # Определяем основные направления с правильными ориентациями
        directions = {
            "bottom": np.array([0.0, -1.0, 0.0]),   # То что раньше было right
            "top": np.array([0.0, 1.0, 0.0]),       # То что раньше было left
            "left": np.array([0.0, 0.0, -1.0]),     # То что раньше было down
            "right": np.array([0.0, 0.0, 1.0])      # То что раньше было up
        }
        
        # Вычисляем перпендикулярные направления
        cross_directions = {}
        for name, dir_vec in directions.items():
            cross = np.cross(to_obstacle_normalized, dir_vec)
            cross_norm = np.linalg.norm(cross)
            if cross_norm > 0:
                cross_directions[name] = cross / cross_norm
        
        # Выбираем направление, ближайшее к направлению на цель
        best_dir = None
        best_dot = -1
        best_name = ""
        
        for name, cross_dir in cross_directions.items():
            dot = np.dot(cross_dir, to_target_normalized)
            if dot > best_dot:
                best_dot = dot
                best_dir = cross_dir
                best_name = name
        
        return best_dir, best_name

    def calculate_avoidance(self, sensor_data: Dict) -> np.ndarray:
        target_dir = self.target_pos - self.drone_pos
        target_dir_norm = np.linalg.norm(target_dir)
        
        if target_dir_norm < 10:
            return np.zeros(3)
        
        if self.navigation_state == "approaching":
            if sensor_data['obstacle_detected']:
                self.navigation_state = "avoiding"
                self.avoid_start_pos = self.drone_pos.copy()
                self.avoid_start_time = time.time()
                
        elif self.navigation_state == "avoiding":
            if self.has_passed_obstacle():
                self.navigation_state = "resuming"
            elif time.time() - self.avoid_start_time > 30:
                self.navigation_state = "resuming"
                
        elif self.navigation_state == "resuming":
            if not sensor_data['obstacle_detected']:
                self.navigation_state = "approaching"
        
        if self.navigation_state == "approaching":
            if target_dir_norm > 0:
                return target_dir / target_dir_norm
            return np.array([1.0, 0.0, 0.0])
            
        elif self.navigation_state == "avoiding":
            avoid_dir, dir_name = self.calculate_avoidance_direction()
            self.avoidance_direction = dir_name  # Сохраняем направление облета
            target_component = (target_dir/target_dir_norm) * 0.6
            avoid_component = avoid_dir * 0.4
            return target_component + avoid_component
            
        elif self.navigation_state == "resuming":
            if target_dir_norm > 0:
                return target_dir / target_dir_norm
            return np.array([1.0, 0.0, 0.0])
        
        return np.array([1.0, 0.0, 0.0])

    def update_position(self, direction: np.ndarray) -> None:
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        
        if self.navigation_state == "avoiding":
            current_speed = self.drone_speed * 0.6
        else:
            current_speed = self.drone_speed * 0.8
        
        self.drone_pos += direction * current_speed
        self.trajectory.append(self.drone_pos.copy())
        
        for i in range(3):
            self.drone_pos[i] = np.clip(self.drone_pos[i], 0, self.world_size[i])

    def project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[int, int]:
        x = int(point_3d[0] * self.display_size[0] / self.world_size[0])
        y = int((self.world_size[1] - point_3d[1]) * self.display_size[1] / self.world_size[1])
        return (x, y)

    def draw_scene(self) -> None:
        self.display_img[:] = 60
        
        # Траектория
        if len(self.trajectory) > 1:
            points_2d = []
            colors = []
            for p in self.trajectory:
                points_2d.append(self.project_3d_to_2d(p))
                color_val = int(255 * (p[2] / self.world_size[2]))
                colors.append((0, color_val, 255 - color_val))
            
            for i in range(len(points_2d)-1):
                cv2.line(self.display_img, points_2d[i], points_2d[i+1], colors[i], 2)
        
        # Препятствие и зоны
        obst_2d = self.project_3d_to_2d(self.obstacle_pos)
        radius_2d = int(self.obstacle_radius * self.display_size[0] / self.world_size[0])
        safety_radius_2d = int((self.obstacle_radius + self.safety_margin) * 
                          self.display_size[0] / self.world_size[0])
        
        overlay = self.display_img.copy()
        cv2.circle(overlay, obst_2d, safety_radius_2d, (0, 100, 255), -1)
        cv2.addWeighted(overlay, 0.2, self.display_img, 0.8, 0, self.display_img)
        cv2.circle(self.display_img, obst_2d, radius_2d, (0, 0, 255), -1)
        
        # Дрон и цель
        drone_2d = self.project_3d_to_2d(self.drone_pos)
        cv2.circle(self.display_img, drone_2d, 8, (0, 200, 255), -1)
        
        target_2d = self.project_3d_to_2d(self.target_pos)
        cv2.drawMarker(self.display_img, target_2d, (0, 255, 0), cv2.MARKER_STAR, 20, 2)
        
        # Линия прямой видимости
        cv2.line(self.display_img, self.project_3d_to_2d(self.drone_pos),
                self.project_3d_to_2d((self.target_pos)), (255, 255, 255), 1)
        
        # Информация с указанием направления облета
        info_text = [
            f"State: {self.navigation_state.upper()}",
            f"Avoidance: {self.avoidance_direction.upper()}" if self.navigation_state == "avoiding" else "Avoidance: NONE",
            f"Position: X={self.drone_pos[0]:.1f}, Y={self.drone_pos[1]:.1f}, Z={self.drone_pos[2]:.1f}",
            f"Speed: {self.drone_speed * (0.6 if self.navigation_state == 'avoiding' else 0.8):.1f} m/s",
            f"Distance to target: {np.linalg.norm(self.target_pos - self.drone_pos):.1f}m",
            f"Obstacle distance: {self.sensor_data['closest_distance']:.1f}m",
            f"Start position: [{self.trajectory[0][0]:.0f}, {self.trajectory[0][1]:.0f}, {self.trajectory[0][2]:.0f}]"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(self.display_img, text, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def run(self) -> None:
        last_frame_time = time.time()
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            current_time = time.time()
            frame_count += 1
            
            self.get_sensor_data()
            move_direction = self.calculate_avoidance(self.sensor_data)
            self.update_position(move_direction)
            
            if current_time - last_frame_time >= 0.05:
                self.draw_scene()
                cv2.imshow('Drone 3D Navigation', self.display_img)
                last_frame_time = current_time
            
            if current_time - start_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                cv2.setWindowTitle('Drone 3D Navigation', 
                                 f'Drone 3D Navigation - FPS: {fps:.1f} | Adaptive Avoidance')
                frame_count = 0
                start_time = current_time
            
            key = cv2.waitKey(1)
            if key == 27 or np.linalg.norm(self.target_pos - self.drone_pos) < 10:
                self.running = False
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    simulator = DroneSimulator3D()
    simulator.run()