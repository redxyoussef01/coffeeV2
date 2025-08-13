import cv2

# --- CONFIG ---
VIDEO_PATH = "input1.mp4"  # Path to your video
ROI_X, ROI_Y, ROI_W, ROI_H = 0, 400, 1900, 300  # (x, y, width, height)

# --- VIDEO CAPTURE ---
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI rectangle (green with thickness 2)
    cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)

    # Display frame
    cv2.imshow("ROI Viewer", frame)

    # Press Q to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
