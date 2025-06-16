import cv2
import numpy as np
from threading import Thread, Lock
import time
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class RealUAVProcessor:
    def __init__(self):
        self.lock = Lock()
        self.running = False
        
        # Параметры объекта
        self.object_detected = False
        self.min_distance = float('inf')
        self.object_position = np.zeros(3)
        self.approach_speed = 0.0
        self.surface_area = 0
        
        # Инициализация ROS для лидара
        rospy.init_node('uav_processor', anonymous=True)
        self.point_cloud = None
        self.last_position = None
        self.last_time = time.time()
        
    def point_cloud_callback(self, msg):
        """ROS callback для данных лидара"""
        cloud_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(cloud_gen))
        
        if len(points) > 0:
            with self.lock:
                distances = np.linalg.norm(points, axis=1)
                self.min_distance = np.min(distances)
                self.object_position = np.mean(points, axis=0)
                
                # Расчет скорости
                current_time = time.time()
                dt = current_time - self.last_time
                if dt > 0 and self.last_position is not None:
                    self.approach_speed = np.dot(
                        (self.object_position - self.last_position),
                        self.object_position / np.linalg.norm(self.object_position)
                    ) / dt
                
                self.last_position = self.object_position.copy()
                self.last_time = current_time

    def process_camera(self):
        """Обработка потока с камеры"""
        cap = cv2.VideoCapture('/dev/video0')  # Или другой источник
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Детекция объекта
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            with self.lock:
                if len(contours) > 0:
                    largest = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 100:  # Порог площади
                        self.object_detected = True
                        self.surface_area = cv2.contourArea(largest)
                    else:
                        self.object_detected = False
                else:
                    self.object_detected = False
        
        cap.release()

    def start(self):
        """Запуск системы"""
        self.running = True
        
        # Подписка на данные лидара
        rospy.Subscriber("/lidar/points", PointCloud2, self.point_cloud_callback)
        
        # Запуск потока обработки камеры
        camera_thread = Thread(target=self.process_camera)
        camera_thread.start()
        
        try:
            while self.running and not rospy.is_shutdown():
                # Здесь может быть ваша логика управления
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.running = False
            camera_thread.join()

    def get_object_data(self):
        """Возвращает текущие данные об объекте"""
        with self.lock:
            return {
                'detected': self.object_detected,
                'min_distance': self.min_distance,
                'position': self.object_position,
                'speed': self.approach_speed,
                'surface_area': self.surface_area,
                'timestamp': time.time()
            }

if __name__ == "__main__":
    processor = RealUAVProcessor()
    processor.start()