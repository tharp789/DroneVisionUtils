import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import reprojection_kinematics as rpk

np.set_printoptions(suppress=True, precision=6)

def create_dummy_calibration(file_path):
    """Creates a fake camera calibration file for testing."""
    camera_matrix = np.array([[1000, 0, 960],
                              [0, 1000, 540],
                              [0, 0, 1]], dtype=np.float64)
    distortion = np.zeros((5,), dtype=np.float64)
    np.savez(file_path, camera_matrix=camera_matrix, distortion=distortion)

def test_inbound_point_reprojection_no_zoom():
    reproj = rpk.PointReprojection("test_calib.npz", 1920, 1080)
    # Test with a point in front of the camera
    camera_point = np.array([0.0, 0.0, 5.0])  # X right, Y down, Z forward in camera coordinates
    zoom_level = 0  # No zoom for this test
    pixel, direction, pixel_inbounds = reproj.get_marker_location(camera_point, zoom_level, reproj.original_image_width / 2, reproj.original_image_height / 2)
    assert pixel_inbounds, "Point should be in bounds"
    print(f"Inbound point projection: {pixel}, direction: {direction}")
    visualize_marker(pixel, direction, pixel_inbounds, reproj.original_image_width, reproj.original_image_height)

def test_out_of_bounds_point_reprojection_no_zoom():
    reproj = rpk.PointReprojection("test_calib.npz", 1920, 1080)
    # Test with a point behind the camera
    camera_point = np.array([5.0, 5.0, -5.0])  # X right, Y down, Z backward in camera coordinates
    zoom_level = 0
    pixel, direction, pixel_inbounds = reproj.get_marker_location(camera_point, zoom_level, reproj.original_image_width / 2, reproj.original_image_height / 2)
    assert not pixel_inbounds, "Point should be out of bounds"
    print(f"Out of bounds point projection: {pixel}, direction: {direction}")
    visualize_marker(pixel, direction, pixel_inbounds, reproj.original_image_width, reproj.original_image_height)

def test_drone_to_camera_point_transform_inbounds_no_zoom():
    reproj = rpk.PointReprojection("test_calib.npz", 1920, 1080)
    # Test with a point in front of the camera
    home_in_drone = np.array([0.0, 100.0, 0.0])  # x right, y down, z forward in drone coordinates
    gimbal_pitch = -45  # degrees
    gimbal_yaw = 90   # degrees
    home_in_base = reproj.location_drone_to_location_base(home_in_drone)
    home_in_cam = reproj.location_base_to_location_camera(home_in_base, gimbal_pitch, gimbal_yaw)
    print(f"Home in camera coordinates: {home_in_cam.astype(np.float32)}")

def test_global_to_drone_transform():
    reproj = rpk.PointReprojection("test_calib.npz", 1920, 1080)
    # Test with a drone position and orientation
    drone_lat = 37.7749  # Example latitude
    drone_long = -122.4194
    heading = 270  # Heading in degrees
    home_lat = 38.7749
    home_long = -122.4194
    relative_altitude = 100.0  # Altitude in meters
    home_in_drone = reproj.location_global_to_location_drone(drone_lat, drone_long, heading, home_lat, home_long, relative_altitude)
    print(f"Home in drone coordinates: {home_in_drone.astype(np.float32)}")

def test_full_reprojection_workflow_no_zoom():
    reprojection = rpk.PointReprojection('rgb1080_camera_calibration.npz', 1920, 1080)
    pixel, direction, inbounds = reprojection.get_pixel_from_location(
        drone_lat=37.0,
        drone_long=-122.4194,
        drone_altitude=10000,
        gimbal_pitch=10,
        gimbal_yaw=0,
        zoom_level=0,
        heading=90,
        home_lat=36.9,
        home_long=-122.3194,
        home_altitude=0
    )
    print(f"Pixel: {pixel}, Direction: {direction}, Inbounds: {inbounds}")
    visualize_marker(pixel, direction, inbounds, reprojection.original_image_width, reprojection.original_image_height)

def test_full_reprojection_workflow_with_zoom():
    reprojection = rpk.PointReprojection('rgb1080_camera_calibration.npz', 1920, 1080)
    pixel, direction, inbounds = reprojection.get_pixel_from_location(
        drone_lat=37.0,
        drone_long=-122.4194,
        drone_altitude=10000,
        gimbal_pitch=10,
        gimbal_yaw=0,
        zoom_level=3,
        heading=90,
        home_lat=36.9,
        home_long=-122.3194,
        home_altitude=0
    )
    print(f"Pixel: {pixel}, Direction: {direction}, Inbounds: {inbounds}")
    visualize_marker(pixel, direction, inbounds, reprojection.original_image_width, reprojection.original_image_height)

def visualize_marker(pixel, direction, inbounds, width=1920, height=1080):
    """
    Visualizes the projection of points onto an image canvas.
    Points are projected from camera coordinates to pixel coordinates.
    Out-of-bounds points are marked with arrows pointing towards the center.
    """
    img = np.full((height, width, 3), 200, dtype=np.uint8)  # light gray bg
    cx, cy = width // 2, height // 2  # principal point at image center

    # Draw center cross
    cv2.line(img, (int(cx)-10, int(cy)), (int(cx)+10, int(cy)), (150,150,150), 1)
    cv2.line(img, (int(cx), int(cy)-10), (int(cx), int(cy)+10), (150,150,150), 1)

    if not inbounds:
        draw_arrow_at_border(img, pixel, direction)

    else:
        # Draw green circle at projected pixel
        cv2.circle(img, (int(round(pixel[0])), int(round(pixel[1]))), 6, (0,255,0), -1)
        cv2.putText(img, "Projected Point", (int(round(pixel[0]))+8, int(round(pixel[1]))-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # Display window (press any key to close)
    cv2.imshow("Projection visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_arrow_at_border(img, edge_pt, angle, color=(0,0,255), size=30, thickness=2):
    """
    Draw a filled triangle arrow with its tip at edge_pt pointing toward object
    """

    # Compute the tail of the arrow based on angle and size
    tail_x = edge_pt[0] - size * np.cos(angle)
    tail_y = edge_pt[1] - size * np.sin(angle)
    tail = (int(tail_x), int(tail_y))

    cv2.arrowedLine(img, tail, edge_pt, color, thickness=thickness, tipLength=0.4)
    return img

if __name__ == "__main__":
    # Create dummy calibration file
    if not os.path.exists("test_calib.npz"):
        create_dummy_calibration("test_calib.npz")

    print("=== Running Tests ===")
    # test_inbound_point_reprojection_no_zoom()
    # test_out_of_bounds_point_reprojection_no_zoom()
    # test_drone_to_camera_point_transform_inbounds_no_zoom()
    # test_global_to_drone_transform()
    # test_full_reprojection_workflow_no_zoom()
    test_full_reprojection_workflow_with_zoom()

