import os
import yaml
import cv2
import numpy as np

class MapHandler:

    def __init__(self,
                 maps_folder,
                 map_file) -> None:
        
        self.maps_folder = maps_folder
        self.map_file = map_file
        self.map_points = None

        #if loading from a .pgm file
        self.map_free_points = None

        self.import_map()

        return

    def import_map(self):

        path = os.path.join(self.maps_folder,self.map_file)

        if os.path.exists(path):

            self.load_map_from_file(map_path=path)
        
        else:
            print("could not find map at: {}".format(path))
    
    def load_map_from_file(self,map_path:str):
        map_file_type = map_path.split(".")[-1]
        if map_file_type == "npy":
            self.map_points = np.load(map_path)
            print("loaded map from {}".format(map_path))
        elif map_file_type == "yaml":
            self.map_points = self.load_map_from_yaml(map_path)
            print("loaded map from {}".format(map_path))

            self.map_free_points = self.load_freespace_from_yaml(map_path)
            print("loaded map free space from {}".format(map_path))
        else:
            print("load_map_from_file: unknown map file type")
        
    def load_map_from_yaml(self,map_path):
        #read the yaml config
        with open(map_path,'r') as file:
            config = yaml.safe_load(file)
        
        #get the map file name
        map_name = config["image"]
        map_dir = os.path.dirname(map_path)
        map_path = os.path.join(map_dir,map_name)

        #get key yaml params
        res_m = config["resolution"]
        origin = config["origin"]

        #load the file
        img = cv2.imread(map_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
        img = np.flipud(img)

        #create a mesh grid of the map space
        rows,cols = img.shape
        x_origin,y_origin,z_origin = origin

        x_coords = np.linspace(x_origin,x_origin + res_m * (cols - 1),cols)
        y_coords = np.linspace(y_origin,y_origin + res_m * (rows - 1), rows)

        xx,yy = np.meshgrid(x_coords,y_coords)
        grid = np.vstack([xx.ravel(),yy.ravel()]).T

        #get the points from the map
        mask = (img.ravel() != 255)
        points = grid[mask]

        return points
    
    def load_freespace_from_yaml(self,map_path):
        #read the yaml config
        with open(map_path,'r') as file:
            config = yaml.safe_load(file)
        
        #get the map file name
        map_name = config["image"]
        map_dir = os.path.dirname(map_path)
        map_path = os.path.join(map_dir,map_name)

        #get key yaml params
        res_m = config["resolution"]
        origin = config["origin"]

        #load the file
        img = cv2.imread(map_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img,240,255,cv2.THRESH_BINARY)[1]
        img = np.flipud(img)

        #create a mesh grid of the map space
        rows,cols = img.shape
        x_origin,y_origin,z_origin = origin

        x_coords = np.linspace(x_origin,x_origin + res_m * (cols - 1),cols)
        y_coords = np.linspace(y_origin,y_origin + res_m * (rows - 1), rows)

        xx,yy = np.meshgrid(x_coords,y_coords)
        grid = np.vstack([xx.ravel(),yy.ravel()]).T

        #get the points from the map
        mask = (img.ravel() != 0)
        points = grid[mask]

        return points