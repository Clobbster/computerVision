# https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/
import cv2

cap = cv2.VideoCapture("highway.mp4")

# object detection from stable camera 
object_detector = cv2.createBackgroundSubtractorMOG2() # https://www.geeksforgeeks.org/python-opencv-background-subtraction/#


''' 
    ret is a boolean variable that returns true if the frame is available.
    frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
    https://stackoverflow.com/questions/28773186/what-does-ret-and-frame-mean-here

    

    Per convention, a single standalone underscore is sometimes used as a name to indicate that a variable is temporary or insignificant.
    For example, in the following loop we dont need access to the running index and we can use “_” to indicate that it is just a temporary value:

    for _ in range(32):
    print('Hello, World.')

    https://dbader.org/blog/meaning-of-underscores-in-python
'''

while True:
    ret, frame = cap.read()

    # object detection
    mask = object_detector.apply(frame) # mask allows us to focus only on the portions of the image that interests us. https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

    for contour in contours:
        # calculate area of important items and remove small elements
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2) # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)
    
    key = cv2.waitKey(30) # displays window for the given duration
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()