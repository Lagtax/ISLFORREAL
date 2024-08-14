import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import font, PhotoImage, Label

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

current_string = ""
last_prediction_time = time.time()
last_predicted_character = ""
cap = None

def start_detection():
    global current_string, last_prediction_time, last_predicted_character, cap
    current_string = ""
    last_prediction_time = time.time()
    last_predicted_character = ""
    cap = cv2.VideoCapture(0)
    detect_sign_language()

def detect_sign_language():
    global current_string, last_prediction_time, last_predicted_character, cap

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            if predicted_character != last_predicted_character:
                last_prediction_time = time.time()
                last_predicted_character = predicted_character
            elif time.time() - last_prediction_time > 2:  # Wait for 2 seconds before adding the character to the string
                current_string += predicted_character
                last_predicted_character = ""  # Reset last predicted character

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        else:
            if time.time() - last_prediction_time > 3:  # If no hand is detected for more than 3 seconds, add a space
                current_string += " "
                last_prediction_time = time.time()

        # Display the current string in the bottom-left corner of the window (in black color)
        cv2.putText(frame, current_string, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord(' '):  # Spacebar to clear the current string
            current_string = ""
        elif key == ord('1'):  # Press '1' to close the program
            break

    cap.release()
    cv2.destroyAllWindows()
    show_menu()

def show_menu():
    root = tk.Tk()
    root.title("Sign Language Detection")

    # Set window size
    window_width = 800
    window_height = 600

    # Get the screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the position of the window to center it
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    # Set the position of the window to the center of the screen
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    try:
        # Load and set the background image
        background_image = PhotoImage(file="C:/Users/neera/Downloads/signlogo.jpg")
        background_label = tk.Label(root, image=background_image)
        background_label.place(relwidth=1, relheight=1)
    except Exception as e:
        print(f"Error loading background image: {e}")

    # Create a custom font
    custom_font = font.Font(family="Helvetica", size=16, weight="bold")

    # Create and place the buttons
    start_button = tk.Button(root, text="Start Detection", font=custom_font, width=20, height=2, command=lambda: [root.destroy(), start_detection()])
    start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    exit_button = tk.Button(root, text="Exit", font=custom_font, command=root.quit)
    exit_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    if current_string:
        result_label = tk.Label(root, text="Predicted String: " + current_string, font=custom_font, bg='white')
        result_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    root.mainloop()

show_menu()