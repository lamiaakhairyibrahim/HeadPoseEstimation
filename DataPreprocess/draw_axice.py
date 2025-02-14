
import cv2
import numpy as np
import scipy.io
from math import cos, sin

class Draw:
    def __init__(self , img , yaw , pitch , roll , x_point_center = None  , y_point_center = None , size = 100 ):
            self.img = img 
            self.yaw = yaw
            self.pitch = pitch  
            self.roll = roll   
            self.x_point_center = x_point_center  
            self.y_point_center = y_point_center 
            self.size = size

    def draw_axis(self):
        self.yaw = -self.yaw
        if self.x_point_center != None and self.y_point_center != None :
            self.x_point_center = self.x_point_center
            self.y_point_center = self.y_point_center
        else:
            self.x_point_center = self.img.shape[1] / 2
            self.y_point_center = self.img.shape[0] /2

        # X-Axis pointing to right. drawn in red
        x1 = self.size * (cos(self.yaw) * cos(self.roll)) + self.x_point_center
        y1 = self.size * (cos(self.pitch) * sin(self.roll) + cos(self.roll) * sin(self.pitch) * sin(self.yaw)) + self.y_point_center

        # Y-Axis | drawn in green
        #        v
        x2 = self.size * (-cos(self.yaw) * sin(self.roll)) + self.x_point_center
        y2 = self.size * (cos(self.pitch) * cos(self.roll) - sin(self.pitch) * sin(self.yaw) * sin(self.roll)) + self.y_point_center

        # Z-Axis (out of the screen) drawn in blue
        x3 = self.size * (sin(self.yaw)) + self.x_point_center
        y3 = self.size * (-cos(self.yaw) * sin(self.pitch)) + self.y_point_center

        cv2.line(self.img, (int(self.x_point_center), int(self.y_point_center)), (int(x1),int(y1)),(0,0,255),3) # x => red
        cv2.line(self.img, (int(self.x_point_center), int(self.y_point_center)), (int(x2),int(y2)),(0,255,0),3) # y => green
        cv2.line(self.img, (int(self.x_point_center), int(self.y_point_center)), (int(x3),int(y3)),(255,0,0),2) # z => blue

        return self.img 