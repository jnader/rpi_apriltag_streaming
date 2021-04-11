from imutils.video import VideoStream
from flask import Response, request
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import math
import numpy as np
import cv2
from pupil_apriltags import Detector

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

vs = VideoStream(usePiCamera=1).start()
time.sleep(5.0)
detector = Detector(families='tag36h11', nthreads=2, quad_decimate=3.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

K = np.float32([[614.27570785, 0., 312.01281694], [  0., 612.71902074, 248.9632528 ],[  0., 0., 1. ]]).reshape(-1,3)
distCoeffs = np.float32([[ 3.09492088e-01, -2.18913647e+00, -6.12897025e-04,  2.83006079e-03]])

ref_X, ref_Y, ref_Z = 0, 0, 0
deltaX, deltaY, deltaZ = 0, 0, 0
x, y, z = 0, 0, 0
bool_val = 1
cX, cY = 0, 0

viz_x, viz_y = 0, 0 # For visualization of saved center

@app.route("/")
def index():
	return render_template("index.html")

def rotationMatrixToEulerAngles(R):
	sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
	singular = sy < 1e-6

	if not singular:
		x = math.atan2(R[2,1], R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else:
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0

	return [math.degrees(x), math.degrees(y), math.degrees(z)]

def detect_tag():
	global vs, outputFrame, lock, nb_tags, ref_X, ref_Y, ref_Z, deltaX, deltaY, deltaZ, bool_val, x, y, z, cX, cY, viz_x, viz_y

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=300)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		results = detector.detect(gray, estimate_tag_pose=True, camera_params=[598.3, 602.1, 312.6, 260], tag_size=0.23)
		nb_tags = len(results)

		for r in results:
			(ptA, ptB, ptC, ptD) = r.corners
			ptB = (int(ptB[0]), int(ptB[1]))
			ptC = (int(ptC[0]), int(ptC[1]))
			ptD = (int(ptD[0]), int(ptD[1]))
			ptA = (int(ptA[0]), int(ptA[1]))

			cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
			cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
			cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
			cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

			(cX, cY) = (int(r.center[0]), int(r.center[1]))
			cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

			ref_pt_1 = (viz_x, 0)
			ref_pt_2 = (viz_x, 300)
			ref_pt_3 = (0, viz_y)
			ref_pt_4 = (300, viz_y)

			cv2.line(frame, ref_pt_1, ref_pt_2, (0, 255, 0), 2)
			cv2.line(frame, ref_pt_3, ref_pt_4, (0, 255, 0), 2)

			(x, y, z) = r.pose_t[0][0], r.pose_t[1][0], r.pose_t[2][0]
			if bool_val == 1:
				bool_val = 0
				ref_X = x
				ref_Y = y
				ref_Z = z

			if x - ref_X < 0:
				deltaX = -1
			elif x - ref_X > 0:
				deltaX = 1
			else:
				deltaX = 0

			if y - ref_Y < 0:
				deltaY = -1
			elif y - ref_Y > 0:
				deltaY = 1
			else:
				deltaY = 0

			if z - ref_Z < 0:
				deltaZ = -1
			elif z - ref_Z > 0:
				deltaZ = 1
			else:
				deltaZ = 0

			euler_angles = rotationMatrixToEulerAngles(r.pose_R)
			roll = euler_angles[0]
			pitch = euler_angles[1]
			yaw = euler_angles[2]

		with lock:
			outputFrame = frame.copy()


def generate():
	global outputFrame, lock

	while True:
		with lock:
			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			if not flag:
				continue

		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/tags_feed")
def tags_feed():
	def generate():
		global nb_tags
		yield str(nb_tags)
	return Response(generate(), mimetype='text')

@app.route("/deltaX_feed")
def deltaX_feed():
	def generate():
		global deltaX
		yield str(deltaX)
	return Response(generate(), mimetype = 'text')

@app.route("/deltaY_feed")
def deltaY_feed():
	def generate():
		global deltaY
		yield str(deltaY)
	return Response(generate(), mimetype='text')
	
@app.route("/deltaZ_feed")
def deltaZ_feed():
	def generate():
		global deltaZ
		yield str(deltaZ)
	return Response(generate(), mimetype='text')

@app.route("/reset_feed", methods=['GET','POST'])
def reset_feed():
	global ref_X, ref_Y, ref_Z, x, y, z, viz_x, viz_y, cX, cY

	data = request.form['reset']
	ref_X = x
	ref_Y = y
	ref_Z = z
	viz_x = cX
	viz_y = cY

	return ('', 200)

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())

	t = threading.Thread(target=detect_tag)
	t.daemon = True
	t.start()

	app.run(host=args["ip"], port=args["port"], debug=False, threaded=True, use_reloader=False)

vs.stop()
