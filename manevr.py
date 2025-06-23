import numpy as np
import cv2
from geopy.distance import great_circle

# --- 2. Исходные данные ---
# Эти значения необходимо будет настроить в соответствии с вашим оборудованием и требованиями
num_points = 25  # Расстояние до границ локальной области (м)
Hi_alt_lim = 100.0  # Верхний предел полета по высоте (м)
Low_alt_lim = 5.0  # Нижний предел полета по высоте (м)
Max_distance_linar = 100.0  # Максимальное расстояние сканирования лидара (м)
Max_speed_fly = 54.0  # Максимальная скорость полета БПЛА (км/ч)  (54 км/ч = 15 м/с)
Video_angleX = 60  # Угол обзора видеокамеры по оси Х (градусы)  (Пример)
Video_angleY = 45  # Угол обзора видеокамеры по оси Y (градусы)  (Пример)
d_len = 6.0  # Ширина матрицы камеры в мм  (Пример)
h_len = 4.5  # Высота матрицы камеры в мм  (Пример)
focus = 3.0  # Эффективное фокусное расстояние объектива видеокамеры в мм  (Пример)
Cam_pixelX = 4056  # Кол-во пикселей видеокамеры по оси Х
Cam_pixelY = 3040  # Кол-во пикселей видеокамеры по оси Y
Cam_diagonal = 7.5  # Размер матрицы видеокамеры по диагонали в мм  (Пример)
Time_cicleCtr = 0.1  # Период выработки управляющего воздействия для БПЛА (с)  (Пример: 100 мс)
DeltaSmax = (Max_speed_fly * 1000 / 3600) * Time_cicleCtr  # Расстояние Smax (м) при перемещении на максимальной скорости за время tупр
SpeedMax = Max_speed_fly  # Текущая скорость БПЛА (км/ч)
Mashtab = 10  # Точность выбора направления полета (Размер сетки LidarDrive и ScreenDrive)  (Пример: 10x10)
AltPrioritet = [5, 6, 4, 7, 9, 3, 8, 2, 1]  # Вектор приоритетов в выборе направлений движения БПЛА (начиная с 1)


# --- 3. Алгоритм выбора управляющего воздействия ---

def calculate_azimuth(phi1, ro1, phi_prev, ro_prev):
    """
    Расчет азимута (геодезического) движения БПЛА (формула 1)
    """
    delta_lambda = ro_prev - ro1
    azimuth = np.arctan2(np.sin(delta_lambda) * np.cos(phi1),
                         np.cos(phi_prev) * np.sin(phi1) - np.sin(phi_prev) * np.cos(phi1) * np.cos(delta_lambda))
    return azimuth


def calculate_distance(phi1, ro1, phi_prev, ro_prev):
    """
    Расчет расстояния между GPS координатами (геодезический).
    Использование geopy.distance.great_circle (формула 2)
    """
    GPS_pred = (np.degrees(phi_prev), np.degrees(ro_prev))  # Преобразование в градусы для geopy
    GPS_current = (np.degrees(phi1), np.degrees(ro1))       # Преобразование в градусы для geopy
    deltaS = great_circle(GPS_pred, GPS_current).km * 1000  # Расстояние в метрах
    return deltaS


def calculate_current_speed(deltaS, time_cicle_ctr):
    """
    Расчет текущей скорости БПЛА (формула 3)
    """
    return deltaS / time_cicle_ctr


def calculate_next_gps(phi1, ro1, azimuth, deltaS):
    """
    Расчет GPS координаты предполагаемого перемещения БПЛА (формулы 2-4)
    """
    # Конвертация в радианы
    phi1_rad = np.radians(phi1)
    ro1_rad = np.radians(ro1)
    azimuth_rad = azimuth  # уже в радианах

    # Расчет по формулам (2-4)
    phi2_rad = np.arcsin(np.sin(phi1_rad) * np.cos(deltaS / 6371000) +
                         np.cos(phi1_rad) * np.sin(deltaS / 6371000) * np.cos(azimuth_rad))
    ro2_rad = ro1_rad + np.arctan2(np.sin(azimuth_rad) * np.sin(deltaS / 6371000) * np.cos(phi1_rad),
                                   np.cos(deltaS / 6371000) - np.sin(phi1_rad) * np.sin(phi2_rad))

    # Конвертация обратно в градусы
    phi2 = np.degrees(phi2_rad)
    ro2 = np.degrees(ro2_rad)

    return phi2, ro2


def project_to_xz(points, lidar_position, max_distance):
    """
    Формирование 2D проекции на плоскость XZ из облака точек (XYZ) (п. 3.1.2).
    """
    points = np.asarray(points)
    distances = np.linalg.norm(points - lidar_position, axis=1)
    # Фильтруем точки в радиусе Smax
    valid_points = points[distances <= max_distance]

    # Преобразование координат для проекции XZ. Смещение на num_points
    xz_projection = np.zeros((2 * num_points, 2 * num_points), dtype=np.uint8)
    for point in valid_points:
        x, _, z = point
        # Масштабирование координат в пределах [0, 2 * num_points]
        x_index = int(x + num_points)
        z_index = int(z + num_points)
        # Проверка на выход за границы массива (защита от ошибок)
        if 0 <= x_index < 2 * num_points and 0 <= z_index < 2 * num_points:
            xz_projection[z_index, x_index] = 1  # Отмечаем наличие точки

    return xz_projection


def determine_lidar_drive(xz_projection, mashtab, num_points):
    """
    Определение "Разрешенных" направлений движения БПЛА по проекции XZ (п. 3.1.3).
    """
    lidar_drive = np.zeros((mashtab, mashtab), dtype=np.uint8)
    step = (2 * num_points) / mashtab
    for i in range(mashtab):
        for j in range(mashtab):
            # Определение границ ячейки в проекции XZ
            x_start = int(j * step)
            x_end = int((j + 1) * step)
            z_start = int(i * step)
            z_end = int((i + 1) * step)
            # Проверка наличия ненулевых точек в ячейке
            if np.any(xz_projection[z_start:z_end, x_start:x_end]):
                lidar_drive[i, j] = 1  # Препятствие в ячейке
    # Определение разрешенных направлений (0 - разрешено, 1 - запрещено)
    direct_drive_lidar = np.zeros(9, dtype=np.uint8)  # 9 направлений
    # Пример:  Предполагаем, что направления соответствуют ячейкам LidarDrive.
    #           Например, направление 5 (центр) соответствует LidarDrive[mashtab//2, mashtab//2].
    #           В реальной реализации нужно определить соответствие между направлениями и сеткой.
    #           Данный код - упрощенный пример.  Нужно адаптировать под вашу сетку и направления.
    for i in range(9):
        #  Предполагаем, что направление 5 (центр)  - всегда разрешено.
        if i == 4:  # Направление 5 - всегда разрешено
            direct_drive_lidar[i] = 0
            continue
        # Пример: определяем, разрешено ли движение по направлению.  Это очень упрощенно.
        #  Нужно правильно привязать направления к LidarDrive.
        row = i % 3  # Пример: ряды 0, 1, 2
        col = i // 3  # Пример: колонки 0, 0, 0, 1, 1, 1, 2, 2, 2
        if lidar_drive[row * (mashtab // 3), col * (mashtab // 3)] == 0:
            direct_drive_lidar[i] = 0  # Разрешено
        else:
            direct_drive_lidar[i] = 1  # Запрещено
    return direct_drive_lidar


def determine_screen_drive(image, mashtab, cam_pixel_x, cam_pixel_y):
    """
    Определение "Разрешенных" направлений движения БПЛА по данным с видеокамеры (п. 3.1.4).
    (Имитация обработки данных с камеры, требует интеграции с cv2)
    """
    # Пример имитации обработки изображения и обнаружения препятствий (прямоугольные области)
    # В реальной реализации нужно использовать cv2 для обработки изображения.
    # Предположим, что мы получили координаты прямоугольных областей препятствий.
    # Здесь просто создаем пример, заполняя некоторые элементы.
    screen_drive = np.zeros((mashtab, mashtab), dtype=np.uint8)
    # Пример:  Определяем прямоугольные области препятствий (в пикселях)
    obstacle_rects = [
        (cam_pixel_x // 4, cam_pixel_y // 4, cam_pixel_x // 2, cam_pixel_y // 2),  # Пример 1
        (cam_pixel_x // 1.5, cam_pixel_y // 3, cam_pixel_x // 2, cam_pixel_y // 2),  # Пример 2
    ]

    # Формируем "виртуальную" сетку (аналогично LidarDrive)
    x_step = cam_pixel_x / mashtab
    y_step = cam_pixel_y / mashtab

    for i in range(mashtab):
        for j in range(mashtab):
            # Определяем границы ячейки сетки
            x_start = int(j * x_step)
            x_end = int((j + 1) * x_step)
            y_start = int(i * y_step)
            y_end = int((i + 1) * y_step)

            # Проверяем, попадает ли ячейка в прямоугольники препятствий
            for rect in obstacle_rects:
                x, y, w, h = rect
                if (x <= x_end and x_start <= x + w and
                        y <= y_end and y_start <= y + h):
                    screen_drive[i, j] = 1  # Препятствие в ячейке

    # Определение разрешенных направлений (0 - разрешено, 1 - запрещено)
    direct_drive_camera = np.zeros(9, dtype=np.uint8)
    #  Пример -  Нужно адаптировать под вашу сетку и направления.
    for i in range(9):
        #  Предполагаем, что направление 5 (центр)  - всегда разрешено.
        if i == 4:  # Направление 5 - всегда разрешено
            direct_drive_camera[i] = 0
            continue
        # Пример: определяем, разрешено ли движение по направлению.  Это очень упрощенно.
        #  Нужно правильно привязать направления к LidarDrive.
        row = i % 3  # Пример: ряды 0, 1, 2
        col = i // 3  # Пример: колонки 0, 0, 0, 1, 1, 1, 2, 2, 2
        if screen_drive[row * (mashtab // 3), col * (mashtab // 3)] == 0:
            direct_drive_camera[i] = 0  # Разрешено
        else:
            direct_drive_camera[i] = 1  # Запрещено
    return direct_drive_camera


def select_allowed_directions(direct_drive_lidar, direct_drive_camera):
    """
    Выбор "Разрешенных" направлений движения БПЛА по совпадению результатов анализа данных. (п. 3.1.5)
    """
    # Побитовое AND для определения общих разрешенных направлений.
    direct_drive_bpla = np.logical_and(direct_drive_lidar == 0, direct_drive_camera == 0)
    no_direct = np.all(direct_drive_bpla == 1)  # Если все направления запрещены

    return direct_drive_bpla, no_direct


def determine_optimal_direction(direct_drive_bpla, phi1, ro1, phi_k, ro_k, time_cicle_ctr, kren_angles, tangazh_angles, alt_prioritet, deltaS, current_speed):
    """
    Определение "Компромиссного" направления движения БПЛА по критерию оптимальности (п. 3.1.6).
    """
    # Список разрешенных направлений
    allowed_directions = np.where(direct_drive_bpla)[0]  # Возвращает индексы True элементов

    if len(allowed_directions) == 0:
        # Если нет разрешенных направлений, возвращаем что-то по умолчанию (например, текущее положение)
        print("Нет разрешенных направлений.")
        return 0, 0, 0  # Возвращаем 0, 0, 0 для угла азимута, тангажа и акселератора

    # Расчет расстояний до конечной точки для каждого разрешенного направления
    distances = {}
    for direction in allowed_directions:
        # Пример: углы крена и тангажа (нужно предоставить таблицу с соответствиями)
        # Проверка наличия ключа в словарях
        if direction not in kren_angles or direction not in tangazh_angles:
            print(f"Предупреждение: Отсутствуют углы для направления {direction}. Пропускаем.")
            continue

        kren = kren_angles[direction]
        tangazh = tangazh_angles[direction]

        #  Расчет азимута направления (нужно уточнить, как определяется направление движения)
        #  Этот расчет требует  интеграции с системой управления БПЛА (PX4, ArduPilot и т.п.)
        #  Здесь -  просто пример.
        azimuth_correction = np.radians(kren)  # Предполагаем, что крен влияет на азимут
        azimuth = azimuth_correction  #  В реальной системе -  более сложный расчет.

        # Расчет GPS координаты предполагаемого перемещения
        phi2, ro2 = calculate_next_gps(phi1, ro1, azimuth, deltaS)
        # Расчет расстояния
        distance = calculate_distance(np.radians(phi2), np.radians(ro2), np.radians(phi_k), np.radians(ro_k))
        distances[direction] = distance

    # Выбор оптимального направления
    if len(distances) > 0:
        min_distance = min(distances.values())
        candidates = [k for k, v in distances.items() if v == min_distance]

        if len(candidates) > 1:
            # Если несколько кандидатов с одинаковым расстоянием, используем приоритеты
            for priority in alt_prioritet:
                if priority -1 in candidates:  # AltPrioritet начинается с 1, а candidates с 0
                    optimal_direction = priority - 1
                    break
            else:
                optimal_direction = candidates[0]  # Если приоритеты не помогли, выбираем первый
        else:
            optimal_direction = candidates[0]

        # Получение углов крена и тангажа для оптимального направления (из таблицы)
        optimal_kren = kren_angles[optimal_direction]
        optimal_tangazh = tangazh_angles[optimal_direction]
        # Расчет акселератора (пример)
        accelerator = 1.0 # Максимальный акселератор - для примера,  в реальной системе -  нужно  рассчитывать.

        print(f"Выбрано направление: {optimal_direction+1}") # Вывод направления
        return np.degrees(azimuth), optimal_tangazh, accelerator  #  Возврат
    else:
         print("Не удалось определить оптимальное направление.")
         return 0, 0, 0

# --- Главная функция алгоритма ---
def obstacle_avoidance_algorithm(lidar_points, image,  phi_prev, ro_prev, phi_1, ro_1, phi_k, ro_k):
    """
    Основная функция алгоритма обхода препятствий.
    """
    # Вывод исходных данных
    print("Исходные данные:")
    print(f"  Текущие координаты: Широта = {phi_1}, Долгота = {ro_1}")
    print(f"  Конечные координаты: Широта = {phi_k}, Долгота = {ro_k}")

    # 1. Расчет расстояния deltaS и текущей скорости
    deltaS = calculate_distance(np.radians(phi_1), np.radians(ro_1), np.radians(phi_prev), np.radians(ro_prev)) # м
    current_speed = calculate_current_speed(deltaS, Time_cicleCtr)  # м/с

    #  Преобразование углов в радианы
    phi_1_rad = np.radians(phi_1)
    ro_1_rad = np.radians(ro_1)

    # 2. Формирование 2D проекции (3.1.2)
    xz_projection = project_to_xz(lidar_points, np.array([0, 0, 0]), DeltaSmax)  # Лидар находится в точке (0,0,0)

    # 3. Определение разрешенных направлений (3.1.3, 3.1.4)
    direct_drive_lidar = determine_lidar_drive(xz_projection, Mashtab, num_points)
    direct_drive_camera = determine_screen_drive(image, Mashtab, Cam_pixelX, Cam_pixelY)

    # 4. Выбор разрешенных направлений (3.1.5)
    direct_drive_bpla, no_direct = select_allowed_directions(direct_drive_lidar, direct_drive_camera)

    # 5. Определение оптимального направления (3.1.6)
    #  Определение углов крена и тангажа (Таблица 2) -  нужно определить соответствие номеров направления и углов.
    #  Это пример!  Нужно адаптировать под вашу систему управления.
    kren_angles = {
        0: -45, 1: 0, 2: 45, 3: 45, 4: 45, 5: 0, 6: -45, 7: 45, 8: 0
    }
    tangazh_angles = {
        0: 45, 1: 45, 2: 45, 3: 0, 4: -45, 5: -45, 6: 45, 7: 0, 8: 0
    }

    azimuth, tangazh, accelerator = determine_optimal_direction(direct_drive_bpla, phi_1, ro_1, phi_k, ro_k, Time_cicleCtr, kren_angles, tangazh_angles, AltPrioritet, deltaS, current_speed)

    # Вывод результатов работы алгоритма
    print("\nРезультаты работы алгоритма:")
    print(f"  Угол азимута: {azimuth} градусов")
    print(f"  Угол тангажа: {tangazh} градусов")
    print(f"  Акселератор двигателя: {accelerator}")
    print(f"  Разрешенные направления (Lidar): {direct_drive_lidar}")
    print(f"  Разрешенные направления (Camera): {direct_drive_camera}")
    print(f"  Общие разрешенные направления: {direct_drive_bpla}")

    return azimuth, tangazh, accelerator, direct_drive_lidar, direct_drive_camera

# --- Пример использования ---
if __name__ == "__main__":
    # --- 1. Имитация данных ---
    #  Вместо этого нужно получать данные от лидара и камеры.
    #  Пример: Имитация облака точек лидара
    #  Предположим, что у нас есть список точек в формате (x, y, z)
    #  Эти точки будут использоваться для построения проекции XZ.
    #  Для тестирования -  сгенерируем случайные точки.
    num_points_test = 25  # Размер куба для генерации случайных точек
    lidar_points = np.random.rand(50, 3) * (2 * num_points_test) - num_points_test  # Случайные точки в кубе
    #  Пример:  Имитация изображения с камеры.  В реальной системе -  нужно получать изображение.
    image = np.zeros((Cam_pixelY, Cam_pixelX, 3), dtype=np.uint8) # Пустое изображение
    #  Пример:  Имитация GPS данных
    phi_prev = 55.75  # Широта предыдущей точки (градусы)
    ro_prev = 37.62  # Долгота предыдущей точки (градусы)
    phi_1 = 55.7501  # Широта текущей точки (градусы)
    ro_1 = 37.6201  # Долгота текущей точки (градусы)
    phi_k = 55.76  # Широта конечной точки (градусы)
    ro_k = 37.63  # Долгота конечной точки (градусы)

    # --- 2. Вызов алгоритма ---
    azimuth, tangazh, accelerator, direct_drive_lidar, direct_drive_camera = obstacle_avoidance_algorithm(
        lidar_points, image,  phi_prev, ro_prev, phi_1, ro_1, phi_k, ro_k
    )