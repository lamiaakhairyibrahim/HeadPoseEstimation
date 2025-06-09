from DataPreprocess.preprocess import DataProcessor , NormData
import cv2 
from DataPreprocess.draw_axice import Draw
from model.models import models
import numpy as np
import pandas as pd
from DataPreprocess.queue_1 import Queue
import mediapipe as mp
import joblib

class collect_out:
    def __init__(self , path_data , video_path):
        self.path_data = path_data
        self.video_path = video_path
        self.out()


    def out(self):
        faceModule = mp.solutions.face_mesh
        opject  = DataProcessor(self.path_data )
        x_point , y_point , pitch_label , yaw_label , roll_label , file_name = opject.img_data()
        #print(x_point.shape )
        #print(y_point.shape )
        #print(pitch_label.shape )
        #print(yaw_label.shape )
        #print(roll_label.shape )
        #print(len(file_name) )
        #print(type(x_point))
        opject2 = NormData(x_point , y_point)
        data = opject2.norm()
        data = pd.DataFrame(data)
        #print(data.head())
        pitch_label  = pitch_label.reshape(-1,1)
        yaw_label    =  yaw_label.reshape(-1,1)
        roll_label   = roll_label.reshape(-1,1)
        label = np.concatenate(( pitch_label , yaw_label , roll_label),axis=1)
        labels = pd.DataFrame(label , columns=['pitch','yaw','roll'])
        model = models(data , labels)
        """pitch_model=model.svcpitch()
        yaw_model=model.svcyaw()
        roll_model=model.svcroll()"""
        smoothing = False
        size = 30
        # Create a VideoCapture object and read from input file
        pitch_model= joblib.load(r"D:\my_projects\inprograss\HeadPoseEstimation\head_pose\src\svr_winner_pitch.pkl")
        yaw_model=   joblib.load(r"D:\my_projects\inprograss\HeadPoseEstimation\head_pose\src\svr_winner_yaw.pkl")
        roll_model=  joblib.load(r"D:\my_projects\inprograss\HeadPoseEstimation\head_pose\src\svr_winner_roll.pkl")
        smoothing = False,
        size = 30
        cap = cv2.VideoCapture(self.video_path)
        width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # initializing a list to store the frames   
        img_array = []

        # Check if video read successfully
        if (cap.isOpened()== True): 
                pitch_queue = Queue(max_size = size)
                yaw_queue = Queue(max_size = size)
                roll_queue = Queue(max_size = size)
                # Read until video is completed
                while(cap.isOpened()):   
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret == True:
                        with faceModule.FaceMesh(static_image_mode=True) as face:
                            # processing the image to detect the face and then generating the land marks (468 for each x,y,z).
                            results = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            if results.multi_face_landmarks != None:
                                for face in results.multi_face_landmarks:
                                    # initializing X and Y lists to store the spacial coordinates of the points
                                    X = []
                                    Y = []
                                    # looping over the landmarks to extract x and y
                                    for j,landmark in enumerate(face.landmark):
                                        x = landmark.x
                                        y = landmark.y
                                        # retrieve the true values of x and y
                                        shape = frame.shape 
                                        relative_x = int(x * shape[1])
                                        relative_y = int(y * shape[0])
                                        X.append(relative_x)
                                        Y.append(relative_y)

                                    X = np.array(X)
                                    Y = np.array(Y)
                                    # centering the data arround the point 100
                                    X_center = X - X[100]
                                    Y_center = Y - Y[100]
                                    d = np.linalg.norm(np.array((X[180],Y[180])) - np.array((X[19],Y[19])))
                                    X_norm = X_center/d
                                    Y_norm = Y_center/d
                                    X_norm = X_norm
                                    Y_norm = Y_norm
                                    points = np.hstack([X_norm,Y_norm]).reshape(1,-1)
                                    # predicting the 3 angels to draw the axis on the image
                                    pred_pitch = pitch_model.predict(points)
                                    pred_yaw = yaw_model.predict(points)
                                    pred_roll = roll_model.predict(points)
                                    
                                    if smoothing  == True:
                                        if not pitch_queue.IsFull(): 
                                            pitch_queue.enqueue(pred_pitch)
                                            yaw_queue.enqueue(pred_yaw)
                                            roll_queue.enqueue(pred_roll)
                                        else:
                                            pitch_queue.dequeue()
                                            yaw_queue.dequeue()
                                            roll_queue.dequeue()
                                            pitch_queue.enqueue(pred_pitch)
                                            yaw_queue.enqueue(pred_yaw)
                                            roll_queue.enqueue(pred_roll)

                                        pitch = sum(pitch_queue.queue)/len(pitch_queue.queue)
                                        yaw = sum(yaw_queue.queue)/len(yaw_queue.queue)
                                        roll = sum(roll_queue.queue)/len(roll_queue.queue)
                                        Draw(frame,pitch,yaw,roll,X[1],Y[1])

                                    else:
                                        op = Draw(frame,pred_pitch,pred_yaw,pred_roll,X[1],Y[1])
                                        # appending the result frame to the img_array list
                                        frame = op.draw_axis()
                                        img_array.append(frame)
                    # Break the loop
                    else: 
                        break
        cap.release()  
        # Closes all the frames
        cv2.destroyAllWindows()
        # converting the frames to video
        out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        #--------------------------------------------------------------------
