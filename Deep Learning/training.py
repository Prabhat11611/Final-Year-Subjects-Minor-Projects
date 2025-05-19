import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)

# Create the directory for the dataset if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize the image counter
img_counter = 0

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Increment the image counter
        img_counter += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(f'dataset/image_{img_counter}.jpg', img[y:y+h, x:x+w])

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Stop if 100 images have been captured
    if img_counter >= 100:
        break

# Release the VideoCapture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
