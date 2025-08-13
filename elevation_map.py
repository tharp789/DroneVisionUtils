import rasterio
import rasterio.merge as merge
from rasterio.io import MemoryFile
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import requests
import time
import tempfile

API_KEY="ak_F0IUdl73_Iu1FlDb2qMyApjlS"
DEM_URL = "https://api.gpxz.io/v1/elevation/hires-raster"
MAX_EDGE_LENGTH = np.sqrt(10) - 0.25 # max query size of 10km^2

class ElevationMap:
    def __init__(self, radius_km=MAX_EDGE_LENGTH, logger=None):
        self.radius_km = min(radius_km, MAX_EDGE_LENGTH)  # Ensure radius does not exceed max edge length
        self.logger = logger
        self.dem_available = False
        
    def fetch_elevation_dem(self, curr_lat, curr_long):
        """
        Downloads elevation data from the Open Topo Data API for a specified range.
        
        Parameters:
            radius (float): Radius in km 
        
        Returns:
            str: Path to the downloaded DEM file.
        """

        try:
            radius_m =  self.radius_km * 1000  # Convert km to meters
            left_bbox = self.get_displaced_latlong(curr_lat, curr_long, 0, -radius_m)[1]
            right_bbox = self.get_displaced_latlong(curr_lat, curr_long, 0, radius_m)[1]
            top_bbox = self.get_displaced_latlong(curr_lat, curr_long, radius_m, 0)[0]
            bottom_bbox = self.get_displaced_latlong(curr_lat, curr_long, -radius_m, 0)[0]
            if top_bbox < bottom_bbox:
                top_bbox, bottom_bbox = bottom_bbox, top_bbox  # Ensure top is always greater than bottom
            start_time = time.perf_counter()
            print("Downloading top left corner DEM data") if self.logger  is None else self.logger.info("Downloading top left corner DEM data")
            top_left = self.query_dem_api(left_bbox, curr_long, top_bbox, curr_lat)
            print("Downloading top right corner DEM data") if self.logger  is None else self.logger.info("Downloading top right corner DEM data")
            top_right = self.query_dem_api(curr_long, right_bbox, top_bbox, curr_lat)
            print("Downloading bottom left corner DEM data") if self.logger  is None else self.logger.info("Downloading bottom left corner DEM data")
            bottom_left = self.query_dem_api(left_bbox, curr_long, curr_lat, bottom_bbox)
            print("Downloading bottom right corner DEM data") if self.logger  is None else self.logger.info("Downloading bottom right corner DEM data")
            bottom_right = self.query_dem_api(curr_long, right_bbox, curr_lat, bottom_bbox)
            end_time = time.perf_counter()
            print("Downloaded DEM data in {:.4f} seconds".format(end_time - start_time)) if self.logger  is None else self.logger.info("Downloaded DEM data in {:.4f} seconds".format(end_time - start_time))

            dem_list = [top_left, top_right, bottom_left, bottom_right]
            self.dem = self.merge_dems(dem_list)
            self.dem_available = True

            self.elevation = self.dem.read(1)  # Read the first band
            self.elevation_shape = self.elevation.shape
            self.map_height, self.map_width = self.elevation.shape

            self.long_left = self.dem.bounds.left
            self.long_right = self.dem.bounds.right
            self.lat_top = self.dem.bounds.top
            self.lat_bottom = self.dem.bounds.bottom
            self.center_lat = (self.lat_top + self.lat_bottom) / 2
            self.center_long = (self.long_left + self.long_right) / 2

            self.lat_interval = (self.lat_top - self.lat_bottom) / self.map_height
            self.long_interval = (self.long_right - self.long_left) / self.map_width

            self.lat_axis_deg = np.linspace(self.lat_top, self.lat_bottom, self.map_height)
            self.long_axis_deg = np.linspace(self.long_left, self.long_right, self.map_width)

            self.elevation_interpolator = RegularGridInterpolator(
                (self.lat_axis_deg, self.long_axis_deg), self.elevation, bounds_error=False, fill_value=None, method='linear')
            
            return True
        
        except Exception as e:
            print(f"Error fetching elevation data: {e}")  if self.logger  is None else self.logger.error(f"Error fetching elevation data: {e}")
            self.dem_available = False
            return False
        
    def merge_dems(self, dem_list):
        merged_data, merged_transform= merge.merge(dem_list)
        out_meta =  dem_list[0].meta.copy()
        out_meta.update({
            "height": merged_data.shape[1],
            "width": merged_data.shape[2],
            "transform": merged_transform})
        memfile = MemoryFile()
        with memfile.open(**out_meta) as dataset:
            dataset.write(merged_data)
        dem =  memfile.open()
        return dem
        
    def query_dem_api(self, left_bbox, right_bbox, top_bbox, bottom_bbox):
        """
        Queries the DEM API for elevation data within the specified bounding box.

        Parameters:
            left_bbox (float): Left longitude of the bounding box.
            right_bbox (float): Right longitude of the bounding box.
            top_bbox (float): Top latitude of the bounding box.
            bottom_bbox (float): Bottom latitude of the bounding box.

        Returns:
            np.ndarray: Elevation data as a 2D numpy array.
        """
        params = {
            'bbox_left': left_bbox,
            'bbox_right': right_bbox,
            'bbox_bottom': bottom_bbox,
            'bbox_top': top_bbox,
            "res_m": 1,
            "projection": "latlon",
            "tight_bounds": "true",
            'api-key': API_KEY
        }
        response = requests.get(DEM_URL, params=params)
        if response.status_code == 200:
            dem_data = response.content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tiff') as temp_file:
                temp_file.write(dem_data)
                dem_path = temp_file.name
            dem = rasterio.open(dem_path)
            return dem
        else:
            raise Exception(f"Failed to fetch DEM data: {response.status_code} - {response.text}")
    
    def find_elevation_interp(self, lat, long):
        # Convert latitude and longitude to pixel coordinates
        latlong = np.array([lat, long])
        elevation = self.elevation_interpolator(latlong)
        if elevation is None or np.isnan(elevation):
            raise ValueError(f"Elevation not found for coordinates: lat={lat}, long={long}")
        return elevation[0] if isinstance(elevation, np.ndarray) else elevation
    
    def find_elevation_index(self, lat, long):
        row, col = self.dem.index(long, lat)
        if 0 <= row < self.elevation.shape[0] and 0 <= col < self.elevation.shape[1]:
            elevation = self.elevation[row, col]
            if np.isnan(elevation):
                raise ValueError(f"Elevation not found for coordinates: lat={lat}, long={long}")
            return elevation
        else:
            raise ValueError(f"Coordinates out of bounds: lat={lat}, long={long}")
    
    def get_displaced_latlong(self, lat1, lon1, x, y):
        """
        Gets the the destination lat and lon given a starting lat, lon, and x, y displacement in meters
        """
        # Destination point - http://www.movable-type.co.uk/scripts/latlong.html
        # φ2 = asin( sin φ1 ⋅ cos δ + cos φ1 ⋅ sin δ ⋅ cos θ )
        # λ2 = λ1 + arctan2( sin θ ⋅ sin δ ⋅ cos φ1, cos δ − sin φ1 ⋅ sin φ2 )
        # φ is latitude, λ is longitude, θ is the bearing (clockwise from north),
        # δ is the angular distance d/R; d being the distance travelled, R the earth’s radius
        epsilon = 7 # degree
        if abs(abs(lat1) - 90) < epsilon or abs(lon1) - 180 > epsilon:
            raise ValueError(f"Invalid latitude or longitude: lat={lat1}, long={lon1}")
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)        
        R = 6371000
        d = np.hypot(x, y) / R
        theta = np.arctan2(y, x)
        lat2 = np.arcsin(
            np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(theta)
        )
        lon2 = lon1 + np.arctan2(
            np.sin(theta) * np.sin(d) * np.cos(lat1),
            np.cos(d) - np.sin(lat1) * np.sin(lat2),
        )
        lon2 = (lon2 + 3 * np.pi) % (2 * np.pi) - np.pi

        lat2 = np.degrees(lat2)
        lon2 = np.degrees(lon2)
        return lat2, lon2

    def lat_lon_distances_meters(self, lat1, lon1, lat2, lon2):
        """
        lat, longs are in degrees
        Returns:
        - lat_distance_m (y): north-south distance in meters (difference in latitude)
        - lon_distance_m (x): east-west distance in meters (difference in longitude)
        """
        R = 6371000  # Earth radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        # North-south distance (latitude)
        lat_distance_m = R * dphi

        # East-west distance (longitude, adjusted for latitude)
        lon_distance_m = R * dlambda * np.cos((phi1 + phi2) / 2)

        return lat_distance_m, lon_distance_m
    

if __name__ == "__main__":
    # dem_path = './pittsburgh_dem.tiff'  # Replace with your DEM file path
    dem_path = './nyc_dem.tiff'  # Replace with your DEM file path
    elevation_map = ElevationMap(dem_path)
    elevation_map.plot_elevation()