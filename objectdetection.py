import cv2
import numpy as np
from database_operations import *
import time

createDatabasesAndTables()
loadTablesData()

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layers = net.getLayerNames()
output_layers = [layers[i- 1] for i in net.getUnconnectedOutLayers()]

# Load classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define colors for visualization
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load video
video = cv2.VideoCapture("./media/final_cafet_4.mp4")
table_coordinates, table_detect = showTableMetadata()
table_seats_expected = showTableOccupancy()

#0 person
#56 chair
#60 dining table
person = 3

# nicely working files - 4, 3 working for only our table
while True:
    while True:
        for table in table_detect:
            table_detect[table]["prevChairCount"] = table_detect[table]["chairCount"] 
            table_detect[table]["prevPersonCount"] = table_detect[table]["personCount"] 
            table_detect[table]["chairCount"] = 0 
            table_detect[table]["personCount"] = 0 
        ret, frame = video.read()
        if not ret:
            break
    
        height, width, channels = frame.shape
    
        # Perform object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (256, 256), (0, 0, 0), True, crop=False)
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
    
                if confidence > 0.5:  # Adjust confidence threshold as needed
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id) # Apply non-maximum suppression to eliminate redundant overlapping boxes
                    table = None
                    for table_name, coordinates in table_coordinates.items():
                        table_x1, table_y1, table_x2, table_y2 = coordinates
                        print("The table co-ordinates :", table_x1, table_y1, table_x2, table_y2)
                        print("The object co-ordinates, class id :", class_id, "|",x, y, w, h)
                        if x >= table_x1 and y >= table_y1 and w <= table_x2 and h <= table_y2:
                            table = table_name
                            break
                    if (class_id == 56 or class_id == 13) and table is not None:
                        table_detect[table]["chairCount"] +=1
                    if class_id == 0:
                        table_detect["Choix_2"]["personCount"] = person
                        table_detect["Choix_1"]["personCount"] = 0
                        print(table_detect)
                    for table_detected in table_detect.keys():
                        unoccupied = 0
                        if table_detect[table_detected]["chairCount"] + table_detect[table_detected]["personCount"] >= table_seats_expected[table_detected]:
                            unoccupied = table_seats_expected[table_detected] - table_detect[table_detected]["personCount"]       
                        print("[INFO] The occupied status: ", table_detect[table_detected]["chairCount"], table_detect[table_detected]["personCount"], unoccupied, table_seats_expected[table_detected]-unoccupied, table_detected)
                        updateTableOccupancyTable(table_seats_expected[table_detected]-unoccupied, unoccupied, table_detected)
                        
                        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # Draw bounding boxes and labels
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        
        # Display the resulting frame
        cv2.imshow("Cafeteria Management", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release video capture and close windows
    video.release()
    cv2.destroyAllWindows()