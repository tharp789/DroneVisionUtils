import numpy as np
import cv2
import os
import sys

R = 6371000  # Earth radius in meters

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

# Frame Conventions -------------------------
# Camera Frame convention is x right, y down, z forward
# World Frame convention is ENU
# Lat is y, Long is x, Altitude is z

class PointReprojection:
    def __init__(self, camera_calibration_file_path, 
                 intrinsics_width, 
                 intrinsics_height):
        
        camera_calibration = np.load(camera_calibration_file_path)
        self.calibration_matrix = camera_calibration['camera_matrix']
        self.distortion = camera_calibration['distortion']

        self.original_calibration_matrix = self.calibration_matrix.copy()
        self.original_image_width = intrinsics_width
        self.original_image_height = intrinsics_height

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
    
    def get_camera_to_base_transform(self, yaw, pitch):
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
        cam_to_base_transform = np.eye(4)
        # yaw rotation is around the y axis
        yaw_rot = np.array([[np.cos(yaw), 0, -np.sin(yaw)], 
                            [0, 1, 0], 
                            [np.sin(yaw), 0, np.cos(yaw)]])
        # pitch rotation is around the x axis
        pitch_rot = np.array([[1, 0, 0], 
                              [0, np.cos(pitch), np.sin(pitch)], 
                              [0, -np.sin(pitch), np.cos(pitch)]])

        # rotation order is first * second * third ... for local rotations
        cam_rot = np.dot(pitch_rot, yaw_rot)
        x_offset = CAMERA_HORIZONTAL_OFFSET * np.sin(yaw)
        z_offset = CAMERA_HORIZONTAL_OFFSET * np.cos(yaw)
        y_offset = CAMERA_VERTICAL_OFFSET

        cam_to_base_transform[:3,:3] = cam_rot
        cam_to_base_transform[:3,3] = [x_offset, y_offset, z_offset]

        return cam_to_base_transform

    def location_drone_to_location_base(self, location_drone):
        '''
        Transform the location from the drone frame to the base frame
        '''
        assert len(location_drone) == 3, "Location drone must be a 3D vector"
        location_drone_to_location_base_translation = np.array([0, -DRONE_TO_BASE_HEIGHT_OFFSET, -DRONE_TO_BASE_FORWARD_OFFSET])
        location_base = location_drone + location_drone_to_location_base_translation
        return location_base

    def location_global_to_location_drone(self, drone_lat, drone_long, heading, home_lat, home_long, relative_altitude):
        '''
        Get the relative world location of the drone in the world frame
        '''
        heading_rad = np.radians(360 - heading)

        #TODO: check if the sign of the displacement is correct
        relative_x_world, relative_y_world = self.lat_lon_distances_meters(drone_lat, drone_long, home_lat, home_long)
        heading_rotation = np.array([[np.cos(heading_rad), np.sin(heading_rad), 0],
                                    [-np.sin(heading_rad), np.cos(heading_rad), 0],
                                    [0, 0, 1]])
        drone_to_world = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_drone_to_world = np.dot(drone_to_world, heading_rotation)

        location_world = np.array([relative_x_world, relative_y_world, relative_altitude])
        location_drone = np.dot(rotation_drone_to_world, location_world)
        return location_drone

    def get_pixel_from_location(self, drone_lat, drone_long, drone_altitude, gimbal_pitch, gimbal_yaw, zoom_level, heading, home_lat, home_long, home_altitude):
        '''heading_rad
        Get the pixel location of a latitude and longitude in the image
        
        Parameters:
        lat (float): The latitude of the object
        long (float): The longitude of the object
        altitude (float): The altitude of the drone
        pitch (float): The pitch angle of the gimbal
        yaw (float): The yaw angle of the gimbal in degrees
        heading (float): The heading of the drone in absolute coordinates in degrees
        Returns:
        tuple: The pixel location of the object in the image (u, v)
        '''
        
        # get the location in the world frame
        rel_alt = drone_altitude - home_altitude
        if rel_alt < 0:
            raise ValueError(f"Relative altitude cannot be negative: {rel_alt} (drone_altitude={drone_altitude}, home_altitude={home_altitude})")
        
        if zoom_level < 0 or zoom_level > 6:
            raise ValueError("Zoom level must be between 0 and 6.")            

        location_drone = self.location_global_to_location_drone(drone_lat, drone_long, heading, home_lat, home_long, rel_alt)

        location_base = self.location_drone_to_location_base(location_drone)
        location_camera = self.location_base_to_location_camera(location_base, gimbal_pitch, gimbal_yaw)

        pixel_u, pixel_v = self.get_pixel_from_location_camera(location_camera)
        if pixel_u is None or pixel_v is None:
            return None, None
        
        # apply zoom
        center_u = self.original_image_width / 2
        center_v = self.original_image_height / 2
        pixel_u, pixel_v = self.get_zoomed_pixel_from_original_pixel(pixel_u, pixel_v, zoom_level, center_u, center_v)

        return pixel_u, pixel_v
    
    def location_base_to_location_camera(self, location_base, gimbal_pitch, gimbal_yaw):
        '''
        Get the location of the point in the camera frame from the base frame
        Parameters:
        location_base (numpy.ndarray): The location of the point in the base frame (3,)
        gimbal_pitch (float): The pitch angle of the gimbal in degrees
        gimbal_yaw (float): The yaw angle of the gimbal in degrees
        
        Returns:
        numpy.ndarray: The location of the point in the camera frame (3,)
        '''
        yaw_rad = - np.radians(gimbal_yaw)  # yaw is in the opposite direction as the coordinate frame
        pitch_rad = np.radians(gimbal_pitch)  # pitch is in the same direction as the coordinate frame
        cam_to_base = self.get_camera_to_base_transform(yaw_rad, pitch_rad)
        altitude = location_base[2] + CAMERA_VERTICAL_OFFSET  # Adjust for camera vertical offset
        adjusted_location_base = np.array([location_base[0], location_base[1], altitude])
        # Apply the inverse transformation to get the location in the camera frame
        location_camera = np.dot(cam_to_base[:3, :3], adjusted_location_base) + cam_to_base[:3, 3]

        return location_camera

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
    
    def get_pixel_from_location_camera(self, location_camera):
        '''
        Get the pixel location of a point in the camera frame
        
        Parameters:
        location_camera (numpy.ndarray): The location of the point in the camera frame (3,)
        Returns:
        tuple: The pixel location of the point in the image (u, v)
        '''
        # point is already in camera frame
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        location_camera = location_camera.reshape((1, 3)).astype(np.float32)

        # Project to 2D
        points_2d, _ = cv2.projectPoints(location_camera, rvec, tvec, self.camera_matrix, self.distortion)
        if points_2d is None or len(points_2d) == 0:
            return None, None
        pixel_x, pixel_y = points_2d.ravel()
        pixel_x = int(pixel_x)
        pixel_y = int(pixel_y)

        return pixel_x, pixel_y

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
    
    def get_zoomed_pixel_from_original_pixel(self, orig_pixel_u, orig_pixel_v, zoom_level, center_u, center_v):
        '''
        Get the zoomed pixel location from the original pixel location

        Parameters:
        orig_pixel_u (int): The x pixel location of the object in the original image
        orig_pixel_v (int): The y pixel location of the object in the original image
        zoom_level (int): The zoom level of the image

        Returns:
        int: The x pixel location of the object in the zoomed image
        int: The y pixel location of the object in the zoomed image
        '''
        if zoom_level < 0 or zoom_level > 6:
            raise ValueError("Zoom level must be between 0 and 6.")
        
        scale_factor = 1 + zoom_level
        zoomed_u = center_u + (orig_pixel_u - center_u) * scale_factor
        zoomed_v = center_v + (orig_pixel_v - center_v) * scale_factor
        
        return int(zoomed_u), int(zoomed_v)

    def scale_intrinsics(self, new_width, new_height):
        '''
        Scale the camera intrinsics by a scale factor
        '''
        print(f"Scaling intrinsics from {self.original_image_width}x{self.original_image_height} to {new_width}x{new_height}") if self.logger is None else self.logger.info(f"*********** Scaling intrinsics from {self.original_image_width}x{self.original_image_height} to {new_width}x{new_height}")
        scale_x = new_width / self.original_image_width
        scale_y = new_height / self.original_image_height

        scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        self.calibration_matrix = np.dot(scale_matrix, self.original_calibration_matrix)

    def lat_lon_distances_meters(self, lat1, lon1, lat2, lon2):

        # Reference: https://www.movable-type.co.uk/scripts/latlong.html
        # Using Haversine formula
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        c = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))
        d = R * c

        y = np.sin(dlambda) * np.cos(phi2)
        x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)

        theta = np.atan2(y, x)

        lat_distance_m = d * np.cos(theta)  # North-South distance
        lon_distance_m = d * np.sin(theta)  # East-West distance

        return lat_distance_m, lon_distance_m



        









