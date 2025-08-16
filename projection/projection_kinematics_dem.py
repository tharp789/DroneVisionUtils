import numpy as np
import cv2
import os
import sys

CAMERA_VERTICAL_OFFSET = 0.055 #meters
CAMERA_HORIZONTAL_OFFSET = 0.010 #meters

DRONE_TO_BASE_FORWARD_OFFSET = 0.00 #meters
DRONE_TO_BASE_HEIGHT_OFFSET = 0.032 #meters

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)
sys.path.append(current)
sys.path.append(os.path.join(parent_directory, 'vision'))

from elevation_map import ElevationMap

# Camera Frame convetion is z forward, x right, y down

# World Frame convention is x east, y north, z up

class ProjectionKinematics:
    def __init__(self, camera_calibration_file_path, 
                 intrinsics_width, 
                 intrinsics_height, 
                 logger=None,
                 raytrace_step_size_m=5,
                 max_mapping_distance_m=1000,
                 dem_radius_km=10,
                 drone_to_base_forward_offset_m=DRONE_TO_BASE_FORWARD_OFFSET, 
                 drone_to_base_height_offset_m=DRONE_TO_BASE_HEIGHT_OFFSET):
        
        self.logger = logger
        camera_calibration = np.load(camera_calibration_file_path)
        self.calibration_matrix = camera_calibration['camera_matrix']
        self.distortion = camera_calibration['distortion']

        self.original_calibration_matrix = self.calibration_matrix.copy()
        self.original_image_width = intrinsics_width
        self.original_image_height = intrinsics_height

        self.raytrace_step_size_m = raytrace_step_size_m
        self.max_mapping_distance_m = max_mapping_distance_m
        self.dem_radius_km = dem_radius_km

        self.elevation_map = ElevationMap(radius_km=self.dem_radius_km, logger=self.logger)
        self.dem_available = False

        self.drone_to_base_forward_offset_m = drone_to_base_forward_offset_m
        self.drone_to_base_height_offset_m = drone_to_base_height_offset_m

    def initialize_elevation_map(self, curr_lat, curr_long):
        '''
        Initialize the elevation map for the current location.
        This will download the elevation data from the Open Topo Data API and create an elevation map.

        Parameters:
        curr_lat (float): The current latitude of the drone
        curr_long (float): The current longitude of the drone
        radius_km (float): The radius in kilometers to fetch the elevation data

        Returns:
        ElevationMap: The elevation map object
        '''
        try:
            self.dem_available = self.elevation_map.fetch_elevation_dem(curr_lat, curr_long)
            return self.dem_available
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to initialize elevation map: {e}")
            self.dem_available = False
            self.elevation_map = None
            return False

    def get_base_to_camera_transform(self, yaw, pitch):
        '''
        Get the camera transform from the base to the camera, from on the gimbal angles. 
        Coordinate frame for the camera is z forward, x right, y down.
        Coordinate frame for the vehicle is z forward, x right, y down.

        Parameters:
        yaw (float): The yaw angle of the gimbal
        pitch (float): The pitch angle of the gimbal
        base_location (numpy.ndarray): The location of the base in the world frame (3,) with x,y,z

        Returns:
        numpy.ndarray: The transform from the base to the camera (4,4)
        '''

        # Get the rotation matrix for the gimbal
        cam_transformation = np.eye(4)
        # yaw rotation is around the y axis
        yaw_rot = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        # pitch rotation is around the x axis
        pitch_rot = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])

        # rotation order is first * second * third ... for local rotations
        cam_rot = np.dot(yaw_rot, pitch_rot)
        x_offset = CAMERA_HORIZONTAL_OFFSET * np.sin(yaw)
        z_offset = CAMERA_HORIZONTAL_OFFSET * np.cos(yaw)
        y_offset = CAMERA_VERTICAL_OFFSET

        cam_transformation[:3,:3] = cam_rot
        cam_transformation[:3,3] = [x_offset, y_offset, z_offset]

        return cam_transformation
    
    def ray_drone_to_ray_world(self, ray_drone, heading):
        '''
        Transform the location from the drone frame to the world frame in latitude and longitude
        Drone frame is z forward, x right, y down
        World frame is x east, y north, z up
        '''
        # compass is clockwise angle, converting to counter clockwise angle
        heading_rad = np.radians(360 - heading)
        heading_rotation = np.array([[np.cos(heading_rad), -np.sin(heading_rad), 0], [np.sin(heading_rad), np.cos(heading_rad), 0], [0, 0, 1]])
        # world to drone is a -90 degree rotation around the x axis
        world_to_drone = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        rotation_world_to_drone = np.dot(heading_rotation, world_to_drone)
        ray_world = np.dot(rotation_world_to_drone, ray_drone)
        ray_world = ray_world / np.linalg.norm(ray_world)  # Normalize the ray vector
        return ray_world
    
    def location_drone_to_location_world(self, location_drone, heading, lat, long):
        '''
        Transform the location from the drone frame to the world frame in latitude and longitude
        Drone frame is z forward, x right, y down
        World frame is x east, y north, z up
        '''
        # compass is clockwise angle, converting to counter clockwise angle
        heading_rad = np.radians(360 - heading)
        heading_rotation = np.array([[np.cos(heading_rad), -np.sin(heading_rad), 0], [np.sin(heading_rad), np.cos(heading_rad), 0], [0, 0, 1]])
        # world to drone is a -90 degree rotation around the x axis
        world_to_drone = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        rotation_world_to_drone = np.dot(heading_rotation, world_to_drone)
        location_world_xyz = np.dot(rotation_world_to_drone, location_drone)
        # get the horizontal distance, from x and y components
        distance_m = np.linalg.norm(location_world_xyz[:2])

        new_lat, new_long = self.elevation_map.get_displaced_latlong(lat, long, location_world_xyz[1], location_world_xyz[0])
        return np.array([new_lat, new_long]), np.array([distance_m, location_world_xyz[2]])
    
    def location_base_to_location_drone(self, location_base):
        '''
        Transform the location from the base frame to the drone frame
        '''
        location_drone_to_location_base_translation = np.array([0, self.drone_to_base_height_offset_m, self.drone_to_base_forward_offset_m])
        location_drone = location_base + location_drone_to_location_base_translation
        return location_drone
    
    def ray_trace_to_location(self, ray_world, altitude, lat, long):
        '''
        Ray trace the ray in the world frame to find the location of the object in the world frame

        Parameters:
        ray_world (numpy.ndarray): The ray in the world frame (3,)
        altitude (float): The altitude of the camera
        lat (float): The latitude of the drone
        long (float): The longitude of the drone

        Returns:
        numpy.ndarray: The location of the object in the world frame (3,)
        float: The distance to the object in meters
        '''
        crossed_ground = False
        distance_m = 0.0

        while crossed_ground is False:
            # Get the next point in the ray
            location_world = np.array([ray_world[0] * distance_m, ray_world[1] * distance_m, ray_world[2] * distance_m + altitude])
            # Convert the location to latitude and longitude
            lat_long_query = self.elevation_map.get_displaced_latlong(lat, long, location_world[1], location_world[0])
            if lat_long_query is None:
                return None, None, "Could not find next lat long on ray trace"
            # Get the elevation at the location
            elevation = self.elevation_map.find_elevation_interp(lat_long_query[0], lat_long_query[1])
            if elevation is None:
                return None, None, "Could not find elevation at lat long query"
            # Check if the ray has crossed the ground
            if location_world[2] <= elevation:
                crossed_ground = True
            else:
                distance_m += self.raytrace_step_size_m
            if distance_m > self.max_mapping_distance_m:
                # print(f"Ray trace exceeded max distance of {self.max_mapping_distance_m} m)") if self.logger is None else self.logger.warning(f"Ray trace exceeded max distance of {self.max_mapping_distance_m} m")
                return None, None, "Ray trace exceeded max distance"

        # Get the final location in the world frame
        horizontal_distance_to_target = np.linalg.norm(location_world[:2])
        target_elevation = self.elevation_map.find_elevation_interp(lat_long_query[0], lat_long_query[1])
        relative_elevation = target_elevation - altitude
        return np.array([lat_long_query[0], lat_long_query[1]]), np.array([horizontal_distance_to_target, relative_elevation]), None
    
    def get_lat_long_from_pixel_dem(self, pixel_u, pixel_v, altitude, pitch, yaw, heading, lat, long):
        '''
        Get the latitude and longitude of a pixel in the world frame

        Parameters:
        pixel_u (int): The x pixel location of the object
        pixel_v (int): The y pixel location of the object
        altitude (float): The altitude of the camera
        pitch (float): The pitch angle of the gimbal
        yaw (float): The yaw angle of the gimbal in degrees
        heading (float): The heading of the drone in absolute coordinates in degrees
        lat (float): The latitude of the drone
        long (float): The longitude of the drone

        Returns:
        numpy.ndarray: The latitude and longitude of the object in the world frame (2,)
        '''

        ray_base = self.get_ray_base_from_pixel(pixel_u, pixel_v, pitch, yaw)
        if ray_base is None:
            return None, None
        ray_world = self.ray_drone_to_ray_world(ray_base, heading)
        lat_long, distances_m, status = self.ray_trace_to_location(ray_world, altitude, lat, long)
        return lat_long, distances_m, status
    
    def get_lat_long_from_pixel_flat(self, pixel_u, pixel_v, altitude, pitch, yaw, heading, lat, long):
        '''
        Get the latitude and longitude of a pixel in the world frame

        Parameters:
        pixel_u (int): The x pixel location of the object
        pixel_v (int): The y pixel location of the object
        altitude (float): The altitude of the camera
        pitch (float): The pitch angle of the gimbal
        yaw (float): The yaw angle of the gimbal in degrees
        heading (float): The heading of the drone in absolute coordinates in degrees
        lat (float): The latitude of the drone
        long (float): The longitude of the drone

        Returns:
        numpy.ndarray: The latitude and longitude of the object in the world frame (2,)
        '''

        altitude_base = altitude - self.drone_to_base_height_offset_m
        location_base = self.get_location_base_from_pixel(pixel_u, pixel_v, altitude_base, pitch, yaw)
        if location_base is None:
            return None, None
        location_drone = self.location_base_to_location_drone(location_base)
        location_world, distance_m = self.location_drone_to_location_world(location_drone, heading, lat, long)
        return location_world, distance_m, "Using naive flat assumption for elevation"
    
    def get_location_base_from_pixel(self, pixel_u, pixel_v, altitude, pitch, yaw):
        '''
        Get the 3D location of a pixel in the gimbal base frame 

        Parameters:
        pixel_u (int): The x pixel location of the object
        pixel_v (int): The y pixel location of the object
        depth (float): The depth of the object
        camera_matrix (numpy.ndarray): The camera matrix (3,3)
        camera_transform (numpy.ndarray): The transform from the base to the camera (4,4)

        Returns:
        numpy.ndarray: The 3D location of the object in the world frame (3,)
        '''

        yaw_rad = - np.radians(yaw) # yaw is in the opposite direction as the coordinate frame
        pitch_rad = np.radians(pitch) # pitch is in the same direction as the coordinate frame
        base_to_cam = self.get_base_to_camera_transform(yaw_rad, pitch_rad)

        # get the 3D location of the object in the camera frame
        cam_ray = self.get_3d_ray_from_pixel(pixel_u, pixel_v, self.calibration_matrix, self.distortion)
        # camera pitch is global (with gravity), yaw is local to camera initialization which is aligned with drone heading

        # frame transform from base to cam, is the same as transform from points in cam to points in base (or vectors too)
        # reference: http://www.cs.cmu.edu/afs/cs/academic/class/15494-s24/lectures/kinematics/jennifer-kay-kinematics-2020.pdf
        # reference: https://www.dgp.toronto.edu/~neff/teaching/418/transformations/transformation.html
        ray_in_base_frame_shot_from_cam = np.dot(base_to_cam[:3,:3], cam_ray)

        # if the ray is pointing above the horizon, then the object is not on the ground or too high up
        # if ray_in_base_frame_shot_from_cam[1] < 0:
        #     return None
        camera_altitude = altitude - CAMERA_VERTICAL_OFFSET
        ray_scale = camera_altitude / ray_in_base_frame_shot_from_cam[1]
        location_cam = ray_in_base_frame_shot_from_cam * ray_scale
        location_base = location_cam + base_to_cam[:3,3]
        return location_base
        
    def get_3d_ray_from_pixel(self, pixel_u, pixel_v, camera_matrix, distortion):
        '''
        Get the 3D ray of a pixel in the camera frame. 

        Parameters:
        pixel_u (int): The x pixel location of the object
        pixel_v (int): The y pixel location of the object
        camera_matrix (numpy.ndarray): The camera matrix (3,3)
        distortion (numpy.ndarray): The distortion coefficients (5,)

        Returns:
        numpy.ndarray: The 3D ray of the object in the camera frame (3,)
        '''

        # gets ideal points in camera frame from pixel location
        undistorted_point = cv2.undistortPoints(np.array([[pixel_u, pixel_v]]).astype(np.float32), camera_matrix, distortion).squeeze()
        x, y = undistorted_point
        ray = np.array([x, y, 1.0])
        ray = ray / np.linalg.norm(ray)
        return ray

    def get_ray_base_from_pixel(self, pixel_u, pixel_v, pitch, yaw):
        '''
        Get the 3D location of a pixel in the gimbal base frame 

        Parameters:
        pixel_u (int): The x pixel location of the object
        pixel_v (int): The y pixel location of the object
        depth (float): The depth of the object
        camera_matrix (numpy.ndarray): The camera matrix (3,3)
        camera_transform (numpy.ndarray): The transform from the base to the camera (4,4)

        Returns:
        numpy.ndarray: The 3D location of the object in the world frame (3,)
        '''

        yaw_rad = - np.radians(yaw) # yaw is in the opposite direction as the coordinate frame
        pitch_rad = np.radians(pitch) # pitch is in the same direction as the coordinate frame
        base_to_cam = self.get_base_to_camera_transform(yaw_rad, pitch_rad)

        # get the 3D location of the object in the camera frame
        cam_ray = self.get_3d_ray_from_pixel(pixel_u, pixel_v, self.calibration_matrix, self.distortion)
        # camera pitch is global (with gravity), yaw is local to camera initialization which is aligned with drone heading

        # frame transform from base to cam, is the same as transform from points in cam to points in base (or vectors too)
        # reference: http://www.cs.cmu.edu/afs/cs/academic/class/15494-s24/lectures/kinematics/jennifer-kay-kinematics-2020.pdf
        # reference: https://www.dgp.toronto.edu/~neff/teaching/418/transformations/transformation.html
        ray_in_base_frame_shot_from_cam = np.dot(base_to_cam[:3,:3], cam_ray)
        ray_in_base_frame_shot_from_cam = ray_in_base_frame_shot_from_cam / np.linalg.norm(ray_in_base_frame_shot_from_cam)
        return ray_in_base_frame_shot_from_cam
    
    def get_original_pixel_from_zoomed_pixel(self, zoomed_pixel_u, zoomed_pixel_v, zoom_level, center_u, center_v):
        '''
        Get the original pixel location from the zoomed pixel location

        Parameters:
        zoomed_pixel_u (int): The x pixel location of the object in the zoomed image
        zoomed_pixel_v (int): The y pixel location of the object in the zoomed image
        zoom_factor (float): The zoom factor of the image

        Returns:
        int: The x pixel location of the object in the original image
        int: The y pixel location of the object in the original image
        '''
        if zoom_level < 0 or zoom_level > 6:
            raise ValueError("Zoom level must be between 0 and 6.")
        
        scale_factor = 1 + zoom_level  # Scale from 1x to 7x
        
        orig_u = center_u + (zoomed_pixel_u - center_u) / scale_factor
        orig_v = center_v + (zoomed_pixel_v - center_v) / scale_factor

        return int(orig_u), int(orig_v)
    
    def scale_intrinsics(self, new_width, new_height):
        '''
        Scale the camera intrinsics by a scale factor
        '''
        print(f"Scaling intrinsics from {self.original_image_width}x{self.original_image_height} to {new_width}x{new_height}") if self.logger is None else self.logger.info(f"*********** Scaling intrinsics from {self.original_image_width}x{self.original_image_height} to {new_width}x{new_height}")
        scale_x = new_width / self.original_image_width
        scale_y = new_height / self.original_image_height

        scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        self.calibration_matrix = np.dot(scale_matrix, self.original_calibration_matrix)



        









