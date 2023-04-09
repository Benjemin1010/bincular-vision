import cv2
import numpy as np

# Set up stereo camera
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# Set up stereo rectification maps (replace with your own maps)
R1 = np.load('R1.npy')
R2 = np.load('R2.npy')
P1 = np.load('P1.npy')
P2 = np.load('P2.npy')
Q = np.load('Q.npy')

# Set up object detection model (replace with your own model)
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Set up distance measurement parameters (replace with your own parameters)
focal_length = 600  # in pixels
focal_length_left = 600
focal_length_right = 600

baseline = 0.1  # in meters

while True:
    # Read frames from stereo camera
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right:
        break

    # Rectify stereo frames
    frame_left = cv2.remap(frame_left, R1, P1, cv2.INTER_LINEAR)
    frame_right = cv2.remap(frame_right, R2, P2, cv2.INTER_LINEAR)

    # Compute disparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray_left, gray_right)

    # Normalize disparity map and convert to depth map
    disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth = cv2.reprojectImageTo3D(disparity, Q)

    # Perform object detection on left frame
    height, width, channels = frame_left.shape
    blob = cv2.dnn.blobFromImage(frame_left, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Only detect people
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # Compute distance to objects
    for i in range(len(boxes)):
        # Get coordinates of bounding box
        x, y, w, h = boxes[i]

        # Compute distance to object in left camera
        depth_obj_left = depth[y:y + h, x:x + w, 2]
        depth_obj_left = depth_obj_left[np.logical_and(depth_obj_left > 0, depth_obj_left < 10)]
        distance_left = (focal_length_left * baseline) / depth_obj_left

        # Compute distance to object in right camera
        depth_obj_right = depth[y:y + h, x:x + w, 2]
        depth_obj_right = depth_obj_right[np.logical_and(depth_obj_right > 0, depth_obj_right < 10)]
        distance_right = (focal_length_right * baseline) / depth_obj_right

        # Compute average distance
        distance = (distance_left + distance_right) / 2

        # Draw distance on frame
        cv2.putText(frame_left, f'{distance:.2f} m', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame_left, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display frames
    cv2.imshow('Left', frame_left)
    cv2.imshow('Right', frame_right)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()