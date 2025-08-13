import projection_kinematics as pk
import numpy as np

def test_world_to_drone():
    # Create a projection kinematics object
    projection_kinematics = pk.ProjectionKinematics("camera_calibration.npz")
    
    # Define the world coordinates
    altitude = 10
    
    # Convert the world coordinates to drone coordinates
    transform = projection_kinematics.get_world_at_drone_to_drone_transform(0, altitude)
    assert transform[0, 3] == 0
    assert transform[1, 3] == altitude
    assert transform[2, 3] == 0

def test_location_drone_to_location_world_xy():
    # Create a projection kinematics object
    projection_kinematics = pk.ProjectionKinematics("camera_calibration.npz", 0, 0)
    
    # Define the drone coordinates
    altitude = 10
    heading = 0
    lat = 0
    long = 0
    location_drone = np.array([0, altitude, 1])
    
    # Convert the drone coordinates to world coordinates
    location_world = projection_kinematics.location_drone_to_location_world(location_drone, heading, lat, long)
    assert location_world[0] == lat
    assert location_world[1] == long
    assert location_world[2] == altitude

def test_location_drone_to_location_drone_xy():
    # Create a projection kinematics object
    
    # Define the base coordinates
    altitude = 10
    heading = - np.pi / 2
    lat = 0
    long = 0
    location_base = np.array([-6, 10, 5])
    drone_z_offset = 3
    ans = np.array([8, 6, 0])
    
    projection_kinematics = pk.ProjectionKinematics("camera_calibration.npz", drone_z_offset, 0)
    # Convert the base coordinates to world coordinates
    location_drone = projection_kinematics.location_base_to_location_drone(location_base)
    location_world = projection_kinematics.location_drone_to_location_world(location_drone, heading, lat, long)
    assert location_world[0] == ans[0]
    assert location_world[1] == ans[1]
    assert location_world[2] == ans[2]

def test_lat_long_from_pixel():
    # Create a projection kinematics object
    projection_kinematics = pk.ProjectionKinematics("camera_calibration.npz")
    
    # Define the drone coordinates
    altitude = 0.4
    heading = 0
    lat = 36
    long = -121
    pitch = 0
    yaw = 0

    # Convert the drone coordinates to world coordinates
    lat_long, dist = projection_kinematics.get_lat_long_from_pixel(960, 960, altitude, pitch, yaw, heading, lat, long)
    print(lat_long)

def test_xy_changes_in_latlong():    
    # Convert the drone coordinates to world coordinates
    lat_long = pk.ProjectionKinematics.get_destination(37, -122, 0, 10000)
    print(lat_long)

def test_real_world_location():
    # Create a projection kinematics object
    projection_kinematics = pk.ProjectionKinematics("camera_calibration.npz")
    
    # Define the drone coordinates
    altitude = 20
    heading = 87.12
    lat = 36.811218
    long = -121.1845892
    pitch = -30
    yaw = 52

    x = 0.9466420412063599
    y = 0.5574753284454346
    w = 0.0633535385131836
    h = 0.07271988689899445

    w_px = 1920
    h_px = 1080

    bounding_box = np.array([x * w_px, y * h_px, w * w_px, h * h_px])
    x_px, y_px = np.array([bounding_box[0] + bounding_box[2] // 2, bounding_box[1] + bounding_box[3]])

    # Convert the drone coordinates to world coordinates
    lat_long, dist = projection_kinematics.get_lat_long_from_pixel(x_px, y_px, altitude, pitch, yaw, heading, lat, long)
    print(lat_long) 

if __name__ == "__main__":
    # test_world_to_drone()
    # test_location_drone_to_location_world_xy()
    # test_lat_long_from_pixel()
    # test_xy_changes_in_latlong()
    test_real_world_location()

