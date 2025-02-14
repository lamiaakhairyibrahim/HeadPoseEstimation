import glob
from pathlib import Path
import mediapipe as mp
import scipy.io
import cv2
import numpy as np
"""class DataProcessor:
    def __init__(self,data_path):
        self.data_path = data_path
        self.img_data()
    
    def img_data(self):
        # Modify this path to match the location of your data.
        FileName = sorted([Path(i).stem for i in glob.glob(self.data_path + r"\*.mat")])
        x_point = []
        y_point = []
        labels = []
        file_name = []
        face_mesh = mp.solutions.face_mesh
        for file_n in FileName :
            with face_mesh.FaceMesh(static_image_mode = True) as Face:
                try:
                            # load image
                            img = cv2.imread(self.data_path +r"\\"+file_n+".jpg")
                            # convert imge from BGR to RBG for mediapip
                            img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                            # processing the image to detect the face and then generating the land marks (468 for each x,y,z).
                            results = Face.process(img_rgb)
                            if results.multi_face_landmarks != None :
                                for i in range(len(results.multi_face_landmarks)):
                                    face = results.multi_face_landmarks[i]
                                    # appending the file names where have been detected.
                                    file_name.append(file_n)    
                                    # Initialize empty lists to store x and y coordinates
                                    x = []
                                    y = []
                                    # Iterate through all the landmarks (points) detected on the face
                                    for landmarks in face.landmark:
                                        x_mp = landmarks.x
                                        y_mp = landmarks.y
                                        # Get the shape (dimensions) of the image (height and width)
                                        shape = img_rgb.shape
                                        # Convert the normalized coordinates to pixel values by multiplying by image dimensions
                                        real_x =  int(x_mp * shape[1])  # Multiply by the width of the image
                                        real_y =  int(y_mp * shape[0])  # Multiply by the height of the image
                                        # Append the computed pixel values to the x and y lists
                                        x.append(real_x)
                                        y.append(real_y)

                                    x = np.array(x)
                                    y = np.array(y)
                                    # appending the points of the images in the list of all image points
                                    x_point.append(x)
                                    y_point.append(y)
                                    try:
                                            # loading the mat file to extract the labels (pitch,yaw,roll)
                                            mat_data = scipy.io.loadmat( self.data_path +r"\\"+file_name+".mat")
                                            # extracting the labels 3 angels
                                            mat_label = mat_data['Pose_Para'][0][:3] # to get (rall , yaw , patch)
                                            # appending the 3 angels to labels list
                                            labels.append(mat_label)
                                    except Exception as e :
                                         print(f"error processing in {file_n}")
                                         continue
                except Exception as er:
                         print(f"error when read image {file_n}")        
                         continue

        # converting features and labels to 2D array
        x_point = np.array(x_point)
        y_point = np.array(y_point)
        labels = np.array(labels)
        # the first label (pitch)
        pitch_label = labels[:,0]
        # the first label (yaw)
        yaw_label = labels[:,1]
        # the first label (roll)
        roll_label = labels[:,2]
        return x_point , y_point , pitch_label , yaw_label , roll_label , file_name"""


"""class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.img_data()

    def img_data(self):
        # Get all .mat file names without extension
        FileName = sorted([Path(i).stem for i in glob.glob(self.data_path + r"\*.mat")])

        x_point = []
        y_point = []
        labels = []
        file_name = []

        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Initialize once

        for file_n in FileName:
            try:
                # Load image
                img_path = self.data_path + r"\\" + file_n + ".jpg"
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Error: Could not read image {file_n}.jpg")
                    continue

                # Convert image from BGR to RGB for MediaPipe
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Detect face landmarks
                results = face_mesh.process(img_rgb)

                if results.multi_face_landmarks:
                    for face in results.multi_face_landmarks:
                        file_name.append(file_n)  # Append filename

                        x = []
                        y = []

                        # Get image shape
                        shape = img_rgb.shape

                        for landmarks in face.landmark:
                            real_x = int(landmarks.x * shape[1])  # Width
                            real_y = int(landmarks.y * shape[0])  # Height
                            x.append(real_x)
                            y.append(real_y)

                        x_point.append(np.array(x))
                        y_point.append(np.array(y))

                        try:
                            # Load .mat file
                            mat_path = self.data_path + r"\\" + file_n + ".mat"
                            mat_data = scipy.io.loadmat(mat_path)

                            # Extract pitch, yaw, roll
                            mat_label = mat_data['Pose_Para'][0][:3]
                            labels.append(mat_label)

                        except Exception as e:
                            print(f"Error processing {file_n}.mat: {e}")
                            continue

            except Exception as er:
                print(f"Error when reading image {file_n}.jpg: {er}")
                continue

        # Convert lists to numpy arrays
        x_point = np.array(x_point, dtype=object)
        y_point = np.array(y_point, dtype=object)
        labels = np.array(labels)

        if labels.shape[0] > 0:  # Ensure labels are not empty before slicing
            pitch_label = labels[:, 0]
            yaw_label = labels[:, 1]
            roll_label = labels[:, 2]
        else:
            pitch_label = yaw_label = roll_label = np.array([])

        return x_point, y_point, pitch_label, yaw_label, roll_label, file_name



if __name__ == "__main__":
    processor = DataProcessor(r"D:\my_projects\HeadPoseEstimation\headposses\src\data\archive\AFLW2000")
"""

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        #self.x_point, self.y_point, self.pitch_label, self.yaw_label, self.roll_label, self.file_name = self.img_data()

    def img_data(self):
        # Get all .mat file names without extension
        FileName = sorted([Path(i).stem for i in glob.glob(self.data_path + r"\*.mat")])

        x_point = []
        y_point = []
        labels = []
        file_name = []

        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Initialize once

        for file_n in FileName:
            try:
                # Load image
                img_path = self.data_path + "\\" + file_n + ".jpg"
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Error: Could not read image {file_n}.jpg")
                    continue

                # Convert image from BGR to RGB for MediaPipe
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Detect face landmarks
                results = face_mesh.process(img_rgb)

                if results.multi_face_landmarks:
                    for face in results.multi_face_landmarks:
                        file_name.append(file_n)  # Append filename

                        x = []
                        y = []

                        # Get image shape
                        shape = img_rgb.shape

                        for landmarks in face.landmark:
                            real_x = int(landmarks.x * shape[1])  # Width
                            real_y = int(landmarks.y * shape[0])  # Height
                            x.append(real_x)
                            y.append(real_y)

                        x_point.append(np.array(x))
                        y_point.append(np.array(y))

                        try:
                            # Load .mat file
                            mat_path = self.data_path + "\\" + file_n + ".mat"
                            mat_data = scipy.io.loadmat(mat_path)

                            # Extract pitch, yaw, roll
                            mat_label = mat_data['Pose_Para'][0][:3]
                            labels.append(mat_label)

                        except Exception as e:
                            print(f"Error processing {file_n}.mat: {e}")
                            continue

            except Exception as er:
                print(f"Error when reading image {file_n}.jpg: {er}")
                continue

        # Convert lists to numpy arrays
        x_point = np.array(x_point, dtype=object)
        y_point = np.array(y_point, dtype=object)
        labels = np.array(labels)

        if labels.shape[0] > 0:  # Ensure labels are not empty before slicing
            pitch_label = labels[:, 0]
            yaw_label = labels[:, 1]
            roll_label = labels[:, 2]
        else:
            pitch_label = yaw_label = roll_label = np.array([])

        return x_point, y_point, pitch_label, yaw_label, roll_label, file_name



class NormData:
    def __init__(self , x , y ):
        self.x = np.array(x, dtype=float) 
        self.y = np.array(y, dtype=float)

    def norm(self):
        center_points_x = self.x - self.x[:,100].reshape(-1,1)
        center_points_y = self.y - self.y[:,100].reshape(-1,1)


        x_180 = self.x[:,180].reshape(-1,1)
        x_19  = self.x[:,19].reshape(-1,1)
        y_180 = self.y[:,180].reshape(-1,1)
        y_19  = self.y[:,19].reshape(-1,1)

        try:
            distance = np.linalg.norm(np.hstack((x_19, y_19)) - np.hstack((x_180, y_180)), axis=1).reshape(-1, 1)
        except Exception as e:
            print(f"An error occurred during the distance calculation: {e}")
            return None




        x_norm = center_points_x / distance
        y_norm = center_points_y / distance

        data = np.concatenate((x_norm,y_norm),axis=1)
        return data


    


    