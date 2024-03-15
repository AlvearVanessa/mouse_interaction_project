"""
AI Virtual Mouse
Mon, 07/03/24 15:11H CET
@uthor: maalvear
Source: https://www.computervision.zone/courses/ai-virtual-mouse/
"""

# Download the necessary video
url = 'https://github.com/AssemblyAI-Examples/mediapipe-python/blob/main/walking.mp4?raw=true'
urllib.request.urlretrieve(url, 'walking.mp4')

# Specify the video filename and create a `Pose` object as before
file = 'walking.mp4'
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Create VideoCapture object
    cap = cv2.VideoCapture(file)

    # Raise error if file cannot be opened
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError

    # Get the number of frames in the video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a NumPy array to store the pose data as before
    # The shape is 3x33x144 - 3D XYZ data for 33 landmarks across 144 frames
    data = np.empty((3, len(poselandmarks_list), length))

    # For each image in the video, extract the spatial pose data and save it in the appropriate spot in the `data` array
    frame_num = 0
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        landmarks = results.pose_world_landmarks.landmark
        for i in range(len(mp_pose.PoseLandmark)):
            data[:, i, frame_num] = (landmarks[i].x, landmarks[i].y, landmarks[i].z)

        frame_num += 1

    # Close the video file
    cap.release()