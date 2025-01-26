# **Parking Detection System**

## **Overview**
The Parking Detection System is an advanced computer vision project designed to monitor and manage parking lots in real-time. Utilizing the YOLO (You Only Look Once) object detection model and OpenCV, this system identifies parked cars, calculates the number of free and occupied parking spots, and issues warnings when occupancy exceeds a defined threshold. This solution aims to optimize parking lot management and enhance user experience.

---

## **Features**
- **Real-Time Detection**: Detects parked vehicles in real-time using a video feed.
- **Occupancy Calculation**: Displays the total number of free and occupied parking spots.
- **Warning System**: Issues a warning when the parking lot is more than 50% full.
- **FPS Monitoring**: Shows real-time processing performance (Frames Per Second).
- **Timestamp Display**: Adds a real-time timestamp to each frame for reference.

---

## **YOLO Model**
YOLO (You Only Look Once) is a state-of-the-art object detection model known for its speed and accuracy. In this project:
- The YOLOv8 pre-trained model is used to detect vehicles (class ID: 2).
- The model predicts bounding boxes around detected vehicles in each frame, which are then matched with predefined parking spot coordinates.

---

## **Technologies Used**
- **Python**: Main programming language.
- **YOLOv8**: Object detection framework.
- **OpenCV**: Image and video processing library.
- **NumPy**: For mathematical operations and data manipulation.

---

## **How It Works**
1. **Video Input**: The system reads a video file of a parking lot.
2. **Object Detection**: YOLOv8 detects vehicles in each frame.
3. **Parking Spot Validation**: Detected vehicles are matched with predefined parking spot coordinates to determine occupancy.
4. **Visualization**: 
   - Free spots are highlighted in green.
   - Occupied spots are highlighted in red.
5. **Warning System**: Displays a warning if the parking lot occupancy exceeds 50%.
6. **Data Logging**: The system logs the total number of parked vehicles in the morning and evening into a `.txt` file.

---

## **File Structure**
- **`Main_code.py`**: Main Python script for the Parking Detection System.
- **`parking1.mp4`**: Sample video file for testing the system.
- **`coco.txt`**: Class labels file used by YOLO for object detection.
- **`parking_log.txt`**: Output file logging morning and evening parking data.

---

## **Setup Instructions**
### Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  ```bash
  pip install ultralytics opencv-python numpy
  ```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/username/Parking-Detection-System.git
   cd Parking-Detection-System
   ```
2. Add the YOLOv8 model (`yolov8s.pt`) to the project directory.
3. Place your parking lot video file in the directory and update the `video_path` variable in `Main_code.py` with its file path.
4. Run the script:
   ```bash
   python Main_code.py
   ```

---

## **Outputs**
- Real-time visualization of the parking lot.
- Morning and evening parking data logged in `parking_log.txt`.
- Warnings displayed on the frame when the lot exceeds 50% occupancy.

---

## **Future Enhancements**
- **Vehicle Classification**: Identify the type of parked vehicles (e.g., car, truck, motorcycle).
- **License Plate Recognition**: Integrate OCR for license plate detection.
- **Mobile Integration**: Develop a mobile app for real-time notifications.
- **Cloud Storage**: Save parking data for long-term analysis and reporting.

---

## **Acknowledgments**
This project utilizes the following resources:
- **YOLOv8** by Ultralytics for object detection.
- **OpenCV** for image processing.

---
