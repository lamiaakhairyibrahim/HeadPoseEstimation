# HeadPoseEstimation
## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [Contributors](#contributors)

## introduction
  - The project is to get the head pose from "2D" landmarks, not 3D landmarks. To do so in a classical way, they have to fit the 2D points with a 3D face model, which is considered a fitting technique. Ultimately, it is not a deterministic solution and has problems. Here comes the ML to replace the 3D fitting assumption, the camera parameters, and the projection.
## Dataset
 - The AFLW2000-3D dataset is a widely used benchmark for 3D head pose estimation and facial landmark localization. It contains 2000 real-world facial images, selected from the AFLW dataset, and augmented with precise 3D annotations.
 - ### ✅ Key Features:
     - Total Images: 2000
     - Head Pose Annotations:
     - Yaw (left/right rotation)
     - Pitch (up/down tilt)
     - Roll (side tilt)
     - Facial Landmarks: 68 3D points per image
     - Includes: Large pose variations and occlusions

## Project Workflow
  - **🧠 Landmark Extraction (MediaPipe)**
    - Instead of using the 68 3D facial landmarks provided in the AFLW2000-3D dataset, I utilized MediaPipe Face Mesh to extract 468 facial landmarks per image.

    - Each point includes (x, y) coordinates (ignoring z).

    - The result is a feature vector of 936 dimensions (468 points × 2 coordinates).

    - These features were used to train models for head pose estimation.

    - This approach provides a denser and more expressive facial representation than the original 68-point format, which may improve performance in complex pose and expression scenarios.

  - ### How to use:
  ``` python 
  from your_script_file import bounding

  dataset_path = 'path_to_dataset'
  output_path = 'path_to_output_folder'
  new_size = (width, height)

  bounding(dataset_path, output_path, new_size) 
  ```

- ## how to run

  - **Creation of virtual environments**
  ```
    python -m venv <name of your environment>
  ```
 - **activation of environment**
 ```
   <name of your environment>\Scripts\activate
 ```
 - **Change the directory inside to the environment**
   ```
   cd <name of your environment>
   ```
 - **creat folder in this directory**
   ```
    md src
   ```
 - **Change the directory inside to src**
    ```
     cd src
    ```
- **Colne this repository:**
   ```
   git clone <url of repo >
  ```
 - **install the required dependencies:**
  ```
   pip install -r requirements.txt
  ```
 - **Run the credit_card_fraud_detection.py script:**
 ```
 python main.py <path of data> <path of video>
 ```
