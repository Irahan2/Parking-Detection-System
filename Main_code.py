import cv2
import numpy as np
from ultralytics import YOLO
import datetime

# Load YOLO model
model = YOLO('yolov8s.pt')

# Parking area coordinates
parking_areas = {
    1: [(52, 364), (30, 417), (73, 412), (88, 369)],
    2: [(105, 353), (86, 428), (137, 427), (146, 358)],
    3: [(159, 354), (150, 427), (204, 425), (203, 353)],
    4: [(217, 352), (219, 422), (273, 418), (261, 347)],
    5: [(274, 345), (286, 417), (338, 415), (321, 345)],
    6: [(336, 343), (357, 410), (409, 408), (382, 340)],
    7: [(396, 338), (426, 404), (479, 399), (439, 334)],
    8: [(458, 333), (494, 397), (543, 390), (495, 330)],
    9: [(511, 327), (557, 388), (603, 383), (549, 324)],
    10: [(564, 323), (615, 381), (654, 372), (596, 315)],
    11: [(616, 316), (666, 369), (703, 363), (642, 312)],
    12: [(674, 311), (730, 360), (764, 355), (707, 308)],
}

# Open video file
video_path = r"C:\Users\caner\OneDrive\Desktop\Python\AI Project\parking1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Video file could not be opened. Please check the path.")
    exit()

paused = False  # Pause state
fps_start_time = datetime.datetime.now()
frame_count = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame (optional)
        frame = cv2.resize(frame, (1020, 500))

        try:
            # Perform predictions
            results = model.predict(frame)
            detections = results[0].boxes.data.cpu().numpy()  # Convert tensor to numpy
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            continue

        free_spots = len(parking_areas)
        occupied_spots = []

        # Check detected vehicles
        for detection in detections:
            try:
                x1, y1, x2, y2, _, class_id = detection
                if int(class_id) == 2:  # Class ID 2 -> Car
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    for spot_id, area in parking_areas.items():
                        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                        if result >= 0:  # Is the vehicle in the parking area?
                            free_spots -= 1
                            occupied_spots.append(spot_id)
                            cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2)  # Red for occupied
                            break
            except Exception as e:
                print(f"Detection error: {e}")
                continue

        # Highlight free spots
        for spot_id, area in parking_areas.items():
            if spot_id not in occupied_spots:
                cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)  # Green for free

        # Display parking information
        cv2.putText(frame, f"Free Spots: {free_spots}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Occupied Spots: {len(occupied_spots)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Spots: {len(parking_areas)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Calculate and display the occupancy rate
        total_spots = len(parking_areas)
        occupied_ratio = len(occupied_spots) / total_spots

        if occupied_ratio >= 0.5:  # Warning if occupancy rate is 50% or higher
            warning_message = f"Warning: Parking is {int(occupied_ratio * 100)}% full!"
            cv2.putText(frame, warning_message, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(warning_message)

        # Display date and time
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, now, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # FPS calculation
        frame_count += 1
        fps_end_time = datetime.datetime.now()
        time_diff = (fps_end_time - fps_start_time).total_seconds()
        if time_diff > 0:
            fps = frame_count / time_diff
            cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show the frame
    cv2.imshow("Parking Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Pause/Resume with 'p'
        paused = not paused
    elif key == 27:  # Exit with ESC
        break

cap.release()
cv2.destroyAllWindows()
