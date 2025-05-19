import face_recognition
import face_rec
import cv2

# Load the known face encodings and names
known_face_encodings = []
known_face_names = []

# Load the training data
for file in os.listdir("training_data"):
    img = face_recognition.load_image_file(f"training_data/{file}")
    img_encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(img_encoding)
    known_face_names.append(file.split(".")[0])

# Initialize the video capture device
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the frame
    faces = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, faces)

    # Loop through each face in the frame
    for face_encoding in face_encodings:
        # Calculate the distances between the unknown face encoding and the known face encodings
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Print the distances
        print("Distances:", distances)

        # Find the index of the minimum distance
        min_distance_index = np.argmin(distances)

        # Check if the minimum distance is below the threshold
        if distances[min_distance_index] < 0.5:
            name = known_face_names[min_distance_index]
        else:
            name = "Unknown"

        # Draw a box around the face
        top, right, bottom, left = faces[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()