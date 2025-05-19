import cv2
import os
import numpy as np

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def capture_faces():
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    name = input("Enter your name: ")
    user_id = input("Enter a unique ID: ")
    dataset_path = os.path.join("dataset", f"{name}_{user_id}")
    assure_path_exists(dataset_path)

    count = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured face image
            cv2.imwrite(os.path.join(dataset_path, f"user.{user_id}.{count}.jpg"), gray[y:y + h, x:x + w])

            cv2.imshow('Face Detection', img)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()