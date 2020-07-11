# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
#from pyimagesearch.motion_detection import SingleMotionDetector
from __future__ import print_function
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import json
import requests
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import numpy as np
import urllib.request
from keras.models import model_from_json

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

addr = 'http://127.0.0.2:5000'
test_url = addr + '/api/test'
content_type = 'image/jpeg'
headers = {'content_type': content_type}


outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)


prototxtPath = 'deploy.prototxt'
weightsPath = 'res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNet(prototxtPath,weightsPath)


# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
	#md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

		



		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        #model code
		(h,w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()

		for i in range(0, detections.shape[2]):

			confidence = detections[0,0,i,2]

			if confidence > 0.2:

				box = detections[0,0,i,3:7]*np.array([w,h,w,h])
				(startX, startY, endX, endY) = box.astype("int")

				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				cv2.rectangle(frame, (startX, startY), (endX,endY), (0, 0, 255), 4)
				org = (startX,endY+25)
				font = cv2.FONT_HERSHEY_SIMPLEX
				color = (255,255,255)
				fontScale = 1
				thickness = 2

				frame_crp = frame[startY:endY,startX:endX]
				frame_crp = cv2.cvtColor(frame_crp,cv2.COLOR_BGR2GRAY)
				frame_crp = cv2.resize(frame_crp,(50,50))

				frame_crp = frame_crp.reshape(-1,50,50,1)
				frame_crp = frame_crp/255
				pred = loaded_model.predict(frame_crp.reshape(-1,50,50,1))
				my_list = map(lambda x: x[0], pred)
				pred = list(my_list)[0]

				if pred > 0.3:
					cv2.rectangle(frame,(startX,endY),(endX,endY+30),(0,0,255),-1)
					cv2.putText(frame, 'Mask', org, font,fontScale, color, thickness, cv2.LINE_AA)
				else:
					cv2.rectangle(frame,(startX,endY),(endX,endY+30),(0,0,255),-1)
					cv2.putText(frame, 'No Mask', org, font,fontScale, color, thickness, cv2.LINE_AA)





		# update the background model and increment the total number
		# of frames read thus far
		#md.update(gray)
		total += 1

		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(port=8000, debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
