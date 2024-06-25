# Real-time Instance Segmentation and Object Tracking with YOLOv8n-seg

This project demonstrates real-time instance segmentation and object tracking using YOLOv8n-seg and OpenCV in Python.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Ultralytics (`pip install git+https://github.com/ultralytics/yolov5`)

Ensure you have a camera connected or use a video stream URL for capturing frames.

## Setup

1. Clone the repository:
   ```bash```
  ``` git clone https://github.com/your_username/your_repo.git```
   ```cd your_repo```
 2. Install dependencies:
    ```bash\n   pip install -r requirements.txt\n```
 3. Download the YOLOv8n-seg model:   ```bash\n   wget https://.../yolov8n-seg.pt\n   ```
## Usage
Run the script `object_tracking.py`
```bash\npython object_tracking.py```
Press `q` to quit the application.
## Functionality

- **Initialization**: The YOLO model (`YOLOv8n-seg`) is initialized in a separate thread to optimize startup time.
- **Frame Processing**: Frames from the camera or video stream are continuously read and processed.
- **Object Tracking**: Detected objects are tracked using instance segmentation masks and bounding boxes provided by YOLOv8n-seg.
- **FPS Display**: Real-time FPS (Frames Per Second) is calculated and displayed on the frame along with a crosshair at the center.
## Customization
- Adjust the video capture source by modifying `cap = cv2.VideoCapture(\"http://192.168.0.15:4747/video\")` in `object_tracking.py`.
- Modify the display settings and annotations in the `show_fps` and `add_crosshair` functions as per your requirements.
## Contributing
Feel free to fork this repository, make changes, and submit pull requests. Any contributions you make are **greatly appreciated**.
