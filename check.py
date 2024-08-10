import os
import cv2
import mediapipe as mp


def process_and_show(image_path, mp_drawing):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hands and landmarks
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print(f"Deleted image: {image_path}")
        # Delete the image with no hands detected
        os.remove(image_path)


# Path to your data folder containing subfolders
data_folder = "data"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Iterate through subfolders
for folder_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Checking images in folder: {folder_name}")
        # Iterate through images in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                process_and_show(image_path, mp_drawing)