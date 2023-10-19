# https://www.youtube.com/watch?v=uMzOcCNKr5A

from ultralytics import YOLO
import cv2

# load video
model = YOLO('yolov8n.pt')

video_path = './highway.mp4'
cap = cv2.VideoCapture(video_path)


# read frames
while True:
    ret, frame = cap.read()

    # detect objects
    # track objects
    results = model.track(frame, persist=True)

    # plot results
    frame_ = results[0].plot()

    # visualize
    cv2.imshow('frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break