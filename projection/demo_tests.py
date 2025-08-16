import projection_kinematics as pk
import cv2
import os
import sys
import numpy as np
from time import sleep
from ultralytics import YOLO
import torch

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
siyi_sdk_path = os.path.join(parent_directory, "siyi_sdk")
  
sys.path.append(parent_directory)
sys.path.append(siyi_sdk_path)

from siyi_sdk.siyi_sdk import SIYISDK

def test_bounding_box_projection(altitude):
    siyi_kinematics = pk.ProjectionKinematics("camera_calibration.npz")
    cam = SIYISDK(server_ip="192.168.144.25", port=37260)
    if not cam.connect():
        print("No connection ")
        exit(1)

    cam.requestSetAngles(0, 0)
    sleep(0.5)
    
    model = YOLO("yolov8n.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    rtsp_url = "rtsp://192.168.144.25:8554/main.264"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)

    person_id = 0
    while True:
        ret, frame = cap.read()
        yaw, pitch, _ = cam.getAttitude()
        if not ret:
            print("Error: Could not read frame.")
            exit(1)
        
        result = model(frame, verbose=False)[0]
        for box in result.boxes:
            # if result is a person
            if box.cls == person_id and box.conf > 0.5:
                # get the bounding box
                bounding_box = box.xyxy.cpu().numpy().astype(int)[0]
                # get the pixel location of the bottom center of the bounding box
                pixel = ((bounding_box[0] + bounding_box[2]) // 2, bounding_box[3])
                location = siyi_kinematics.get_location_base_from_pixel(pixel[0], pixel[1], altitude, pitch=pitch, yaw=yaw)
                break
            else:
                location = None

        # visualize the video stream with bounding box and location printed
        if location is not None:
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Location: {location * 39.3701} in", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Bounding Box", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.disconnect()
    
def test_location_accuracy(target_pitch, target_yaw, altitude, expected_distance):
    siyi_kinematics = pk.ProjectionKinematics("camera_calibration.npz", 0, 0)
    cam = SIYISDK(server_ip="192.168.144.25", port=37260)
    if not cam.connect():
        print("No connection ")
        exit(1)

    cam.requestSetAngles(target_yaw, target_pitch)
    sleep(0.5)
    act_yaw, act_pitch, _ = cam.getAttitude()
    print(f"Current Gimbal State -> Yaw: {act_yaw}, Pitch: {act_pitch}")
    sleep(0.5)
    cam.requestFocusHold()

    rtsp_url = "rtsp://192.168.144.25:8554/main.264"

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)
    ret, frame = cap.read()
    cap.release()

    # draw horizontal and vertical lines in the frame
    cv2.line(frame, (0, frame.shape[0]//2), (frame.shape[1], frame.shape[0]//2), (0, 255, 0), 2)
    cv2.line(frame, (frame.shape[1]//2, 0), (frame.shape[1]//2, frame.shape[0]), (0, 255, 0), 2)
    pixel = get_pixel_from_click(frame)
    cv2.circle(frame, pixel, 5, (0, 0, 255), -1)
    cv2.imshow("Selected pixel", frame)
    cv2.waitKey(0)
    location = siyi_kinematics.get_location_base_from_pixel(pixel[0], pixel[1], altitude, pitch=act_pitch, yaw=act_yaw)
    bounding_box = siyi_kinematics.get_xy_bounding_box(pixel[0], pixel[1], altitude, pitch=act_pitch, yaw=act_yaw, pixel_bounding=5)

    estimated_horizontal_distance = np.sqrt(location[0] ** 2 + location[2] ** 2)
    dist_error = np.abs(estimated_horizontal_distance - expected_distance)

    # Reseting the camera
    cam.requestSetAngles(0, 0)
    cam.disconnect()
    sleep(0.5)

    print(f"Location: {location * 39.3701} in")
    print(f"Bounding box z: {bounding_box['min_diff_z'] * 39.3701} in to {bounding_box['max_diff_z'] * 39.3701} in")
    print(f"Bounding box x: {bounding_box['min_diff_x'] * 39.3701} in to {bounding_box['max_diff_x'] * 39.3701} in")
    print(f"Estimated horizontal distance: {estimated_horizontal_distance * 39.3701} in")
    print(f"Horizontal distance error: {dist_error * 39.3701} in")
    cam.disconnect()

def get_pixel_from_click(image):
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param.append((x, y))
            cv2.destroyAllWindows()  # Close the window as soon as the click is registered

    cv2.imshow("Click on the desired pixel for 3D location", image)
    points = []
    cv2.setMouseCallback("Click on the desired pixel for 3D location", on_mouse, points)
    
    while not points:  # Wait until a point is clicked
        if cv2.waitKey(1) & 0xFF == 27:  # Optional: Allow 'Esc' key to exit without selecting a point
            break
    
    return points[0] if points else None

if __name__ == "__main__":
    pitch = 0
    yaw = 0
    altitude_in = 33.5
    altitude_m = altitude_in * 0.0254
    expected_distance_in = 132
    expected_distance_m = expected_distance_in * 0.0254
    print("Testing location accuracy")
    test_location_accuracy(pitch, yaw, altitude_m, expected_distance_m)

    # altitude_in = 33.5
    # # altitude_m = altitude_in * 0.0254
    # altitude_m = 20.0
    # print("Testing bounding box projection")
    # test_bounding_box_projection(altitude_m)


    


    

