import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt', 'yolov8m.pt', etc. for different versions

# Define labels of interest (vehicles) based on YOLOv8 model
class_list = model.names  # Get class names from the model

# Open the video file
video_path = 'small.mp4'  # Ensure this path is correct
cap = cv2.VideoCapture(video_path)

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the red and blue line positions (y-coordinates)
red_line_y = 198
blue_line_y = 268
offset = 7

# Vehicle tracking variables
counter_down = []  # List to count vehicles going down
counter_up = []    # List to count vehicles going up
down = {}          # Dictionary to track vehicles crossing the red line
up = {}            # Dictionary to track vehicles crossing the blue line

tracker = Tracker()
count = 0

# Create a VideoWriter object to save the output video
output_path = 'video.mp4'  # Specify output filename
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()  # Convert to NumPy array
    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        c = class_list[d]
        if 'car' in c or 'truck' in c or 'bus' in c:  # Include other vehicle types if necessary
            list.append([x1, y1, x2, y2])  # Only include the bounding box coordinates

    bbox_id = tracker.update(list)  # Pass only the bounding boxes

    for i, bbox in enumerate(bbox_id):
        x3, y3, x4, y4, id = bbox  # Unpacking the ID and coordinates
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        # Draw bounding box
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw the bounding box

        # Prepare text and calculate size for background
        vehicle_type = class_list[int(px.iloc[i][5])]
        text_size = cv2.getTextSize(vehicle_type, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = x3
        text_y = y3 - 10

        # Draw background rectangle for text
        cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)

        # Display vehicle type
        cv2.putText(frame, vehicle_type, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text

        # Draw the center of the bounding box
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Draw a blue circle at the center

        # Condition for counting vehicles crossing the lines
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):  # Vehicle crosses the red line
            if id not in down:  # Only count if not already counted
                down[id] = cy  # Track the vehicle crossing the red line
                counter_down.append(id)  # Count the vehicle going down

        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):  # Vehicle crosses the blue line
            if id not in up:  # Only count if not already counted
                up[id] = cy  # Track the vehicle crossing the blue line
                counter_up.append(id)  # Count the vehicle going up

    # Draw the lines and counts on the frame
    text_color = (255, 255, 255)  # white color for text
    red_color = (0, 0, 255)  # (B, G, R)   
    blue_color = (255, 0, 0)  # (B, G, R)
    green_color = (0, 255, 0)  # (B, G, R)  

    cv2.line(frame, (172, red_line_y), (774, red_line_y), red_color, 3)
    cv2.putText(frame, 'red line', (172, red_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
    cv2.line(frame, (8, blue_line_y), (927, blue_line_y), blue_color, 3)
    cv2.putText(frame, 'blue line', (8, blue_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    downwards = len(counter_down)
    cv2.putText(frame, 'going down - ' + str(downwards), (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1, cv2.LINE_AA)

    upwards = len(counter_up)
    cv2.putText(frame, 'going up - ' + str(upwards), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Write the processed frame to the output video
    out.write(frame)

    # Show the current frame
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
out.release()  # Finalize and save the video
cv2.destroyAllWindows()
