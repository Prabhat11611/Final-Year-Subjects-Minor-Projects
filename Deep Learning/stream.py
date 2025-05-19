import streamlit as st
import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained model here
# model = ...

def detect_faces(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image

def main():
    st.title("Face Recognition App")

    # Create a sidebar with options
    option = st.sidebar.selectbox("Select an option", ("Home", "Login"))

    if option == "Home":
        st.write("Welcome to the Face Recognition App!")

    elif option == "Login":
        st.write("Please allow camera access to login.")

        # Create a button to start the camera
        if st.button("Start Camera"):
            # Access the camera
            cap = cv2.VideoCapture(0)

            # Capture a frame
            ret, frame = cap.read()

            # Release the camera
            cap.release()

            # Detect faces in the frame
            frame = detect_faces(frame)

            # Display the frame
            st.image(frame, channels="BGR")

if __name__ == "__main__":
    main()
