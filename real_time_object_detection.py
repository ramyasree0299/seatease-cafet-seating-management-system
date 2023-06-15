# How to run?: python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# python real_time.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import packages
from imutils.video import VideoStream, FileVideoStream
from imutils.video import FPS
import numpy as np
import time
import argparse
import imutils
import time
import cv2

# Database code
# ----------------------------
# from test_mysql_connection import mydb
# mycursor = mydb.cursor()
# mycursor.execute("SELECT * FROM table_occupancy")
# myresult = mycursor.fetchall()
# for x in myresult:
# 	print(x)
# ----------------------------

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assigning random colors to each of the classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Live Stream code
# ----------------------------
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).starts()
# ----------------------------


print("[INFO] Video Analysis Started...")
vs = FileVideoStream(path="./media/cafet_6.MOV").start()

time.sleep(2.0)

fps = FPS().start()

objectCount = {
    "people": {
        "currentCount": 0,
        "prevCount": 0
    },
    "chair": {
        "currentCount": 0,
        "prevCount": 0
    }
}

sleepInterval = 10

# Todo: Recieve a notification from application with number of expected chairs. 
expectedChairs = 1
chairsFoundCounterCheck = (60 / sleepInterval)
# Todo: Send an event to mobile to start the 30 mins duration for TABLE XYZ here.
while True:
    objectCount["people"]["prevCount"] = objectCount["people"]["currentCount"]
    objectCount["chair"]["prevCount"] = objectCount["chair"]["currentCount"]
    objectCount["people"]["currentCount"] = 0
    objectCount["chair"]["currentCount"] = 0
    frame = vs.read()
    print("The frame type: ", type(frame))
    if frame is None:
        break
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    resized_image = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image, (1 / 127.5), (300, 300), 127.5, swapRB=True)
    net.setInput(blob)  # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    predictions = net.forward()

    for i in np.arange(0, predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]
        if confidence > args["confidence"]:
            # extract the index of the class label from the 'predictions'
            # idx is the index of the class label
            # E.g. for person, idx = 15, for chair, idx = 9, etc.
            idx = int(predictions[0, 0, i, 1])
            # then compute the (x, y)-coordinates of the bounding box for the object
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            # Example, box = [130.9669733   76.75442174 393.03834438 224.03566539]
            # Convert them to integers: 130 76 393 224
            (startX, startY, endX, endY) = box.astype("int")
            print("The co-ordinates ", (startX, startY, endX, endY))

            # Get the label with the confidence score
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("Object detected: ", label)
            # Draw a rectangle across the boundary of the object
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # Put a text outside the rectangular detection
            # Choose the font of your choice: FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX, FONT_HERSHEY_SCRIPT_COMPLEX, FONT_ITALIC, etc.
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            if idx == 15:
                objectCount["people"]["currentCount"] += 1
            if idx == 9:
                objectCount["chair"]["currentCount"] += 1
    # show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # Todo: Sent event to the mobile application and refresh the table status at the application side.
    print("The number of occupied seats:", objectCount["people"]["currentCount"])
    print("The number of empty seats:", objectCount["chair"]["currentCount"])

    # Check every 5 seconds
    #	time.sleep(sleepInterval)

    if (expectedChairs == objectCount["chair"]["currentCount"]) and (
            expectedChairs == objectCount["chair"]["prevCount"]):
        chairsFoundCounterCheck -= 1
    # Todo: Send an event that the last 1 minute the expected Chairs are available to be occupied.
    else:
        chairsFoundCounterCheck = (60 / sleepInterval)
    if chairsFoundCounterCheck == 0:
        print("Empty table ready to be occupied")
        chairsFoundCounterCheck = (60 / sleepInterval)
    print("The chairs found counter check: ", chairsFoundCounterCheck)
    # Press 'q' key to break the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer
fps.stop()

# Display FPS Information: Total Elapsed time and an approximate FPS over the entire video stream
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# Destroy windows and cleanup
cv2.destroyAllWindows()
# Stop the video stream
vs.stop()

# cd downloads/real-time-object-detection/cafeteria-management-system
# python3 real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# /usr/local/mysql/bin/mysql -uroot -p
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
