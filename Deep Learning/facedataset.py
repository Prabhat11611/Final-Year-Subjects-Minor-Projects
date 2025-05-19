import cv2
import os

def assure_path_exists(path):
    os.makedirs(path, exist_ok=True) 

# Initialize webcam and face detector
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if detector.empty():
    print("Error: Could not load face detection model.")
    exit()

# Get user input (name and ID)
name = input("Enter your name: ")
user_id = input("Enter a unique ID: ")

# Create dataset directory
dataset_path = os.path.join("dataset", f"{name}_{user_id}")
assure_path_exists(dataset_path)

# Image capture loop
count = 0
num_images_to_capture = 100
print("Capturing images... Press 'q' to quit early.")

while count < num_images_to_capture:
    ret, img = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(f"{dataset_path}/face.{str(count)}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow('Capturing Faces', img)
        print(f"Captured {count} images...")

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print(f"Dataset captured for {name} (ID: {user_id}) with {count} images.")
