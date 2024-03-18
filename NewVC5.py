import cv2
import numpy as np  # Import NumPy
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import threading

# Warm-up the camera by reading and discarding a few frames
cap = cv2.VideoCapture("http://192.168.0.15:4747/video")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

for _ in range(10):
    _ = cap.read()

# Initialize the YOLO model outside the main loop
model = None

def initialize_model():
    global model
    model = YOLO("yolov8n-seg.pt")

# Start a thread to initialize the model
model_thread = threading.Thread(target=initialize_model)
model_thread.start()

# Wait for the model thread to finish before entering the main loop
model_thread.join()

# Initialize the timer outside the show_fps function
timer = cv2.getTickCount()

# Initialize variable to store ID of object in the middle
object_in_middle_id = None
object_in_middle_center = [0,0]



# Function to add a crosshair to the image
def add_crosshair(image):
    # Get the center coordinates of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

    # Draw horizontal line
    cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)

    # Draw vertical line
    cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

def show_fps(frame):
    global timer
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    fps_text = f"FPS: {fps:.2f}"

    # Create a blank image for FPS text
    fps_img = np.zeros_like(frame)
    cv2.putText(fps_img, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Call the add_crosshair function
    add_crosshair(fps_img)

    
    
    # cv2.putText(fps_img, "Tracked obj ID: "+ str(object_in_middle_id), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.putText(fps_img, "Tracked obj coord: "+ str(object_in_middle_center[0]) + ", " + str(object_in_middle_center[1]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Overlay FPS text with crosshair on the original frame
    frame_with_fps = cv2.addWeighted(frame, 1, fps_img, 0.5, 0)
    timer = cv2.getTickCount()

    return frame_with_fps

# Continue with the main loop
while True:
    ret, im0 = cap.read()
    # Get the center coordinates of the image outside the loop
    center_x, center_y = im0.shape[1] // 2, im0.shape[0] // 2

    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(track_id, True),
                               track_label=str(track_id))

            

    # Call the show_fps function
    frame_with_fps = show_fps(im0)
    
    cv2.imshow("instance-segmentation-object-tracking", frame_with_fps)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
