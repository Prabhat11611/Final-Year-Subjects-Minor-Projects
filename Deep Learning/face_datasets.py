import cv2
import os

# Function to ensure dataset directory exists
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Initialize the webcam
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ask user for the person's name and ID
person_name = input("Enter the person's name: ")
face_id = input("Enter a numeric face ID (e.g., 1, 2): ")

# Ensure the dataset directory exists
dataset_path = "dataset/"
assure_path_exists(dataset_path)

# Initialize sample face image count
count = 0

# Start capturing images
print("\n[INFO] Initializing face capture. Look at the camera and wait...")

while True:
    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loop for each face found in the frame
    for (x, y, w, h) in faces:
        # Increment sample face image count
        count += 1

        # Save the captured image into the datasets folder with the person's name and ID
        cv2.imwrite(f"{dataset_path}{person_name}.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])

        # Draw rectangle around the face and display the video frame
        cv2.rectangle(image_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Face Capture', image_frame)

    # Press 'q' to stop capturing
    if cv2.waitKey(100) & 0xFF == ord('q'):
        print("\n[INFO] Quitting video capture.")
        break
    # Stop capturing after 100 images
    elif count >= 100:
        print(f"\n[INFO] Captured {count} images, stopping.")
        break

# Release the webcam and close all OpenCV windows
vid_cam.release()
cv2.destroyAllWindows()

print(f"\n[INFO] Successfully captured images for {person_name} with ID {face_id}.")