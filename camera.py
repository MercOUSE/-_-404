import cv2

def get_largest_contour(contours):
    largest_area = 0
    largest_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
            
    return largest_contour, largest_area

def main():
    cap = cv2.VideoCapture(0)  # Захват видео с веб-камеры
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) != 0:
            largest_contour, largest_area = get_largest_contour(contours)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            text = f'Площадь: {int(largest_area)} пикселей'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Frame', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()