import cv2

cap = cv2.VideoCapture("highway.mp4")

# object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    # object detection
    mask = object_detector.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # calculate area of important items and remove small elements
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()