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

vs = VideoStream(usePiCamera=0).start()
time.sleep(5.0)
detector = Detector(families='tag36h11', nthreads=2, quad_decimate=3.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

K = np.float32([[614.27570785, 0., 312.01281694], [  0., 612.71902074, 248.9632528 ],[  0., 0., 1. ]]).reshape(-1,3)
distCoeffs = np.float32([[ 3.09492088e-01, -2.18913647e+00, -6.12897025e-04,  2.83006079e-03]])

ref_ptA, ref_ptB, ref_ptC, ref_ptD = 0, 0, 0, 0
ref_Z = 0
ref_cX, ref_cY = 0, 0
deltaX, deltaY, deltaZ = 0, 0, 0
x, y, z = 0, 0, 0
bool_val = 1
cX, cY = 0, 0

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
	global vs, outputFrame, lock, nb_tags, ref_Z, z, ref_ptA, ptA, ref_ptB, ptB, ref_ptC, ptC, ref_ptD, ptD, ref_cX, cX, ref_cY, cY, deltaX, deltaY, deltaZ, bool_val

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=300)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# results = detector.detect(gray, estimate_tag_pose=True, camera_params=[K[0,0], K[1,1], K[0,2], K[1,2]], tag_size=0.15)
		results = detector.detect(gray, estimate_tag_pose=True, camera_params=[557, 561, 360, 235], tag_size=0.15)
		nb_tags = len(results)

		for r in results:
			# Get tag's corners.
			(ptA, ptB, ptC, ptD) = r.corners
			ptB = (int(ptB[0]), int(ptB[1]))
			ptC = (int(ptC[0]), int(ptC[1]))
			ptD = (int(ptD[0]), int(ptD[1]))
			ptA = (int(ptA[0]), int(ptA[1]))

			# Draw current corners.
			cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
			cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
			cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
			cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

			if bool_val == 1:
				bool_val = 0
				ref_ptA = ptA
				ref_ptB = ptB
				ref_ptC = ptC
				ref_ptD = ptD
				ref_cX  = cX
				ref_cY  = cY

			# Draw desired corners.
			cv2.line(frame, ref_ptA, ref_ptB, (0, 0, 255), 2)
			cv2.line(frame, ref_ptB, ref_ptC, (0, 0, 255), 2)
			cv2.line(frame, ref_ptC, ref_ptD, (0, 0, 255), 2)
			cv2.line(frame, ref_ptD, ref_ptA, (0, 0, 255), 2)

			# Get & draw current tag center.
			(cX, cY) = (int(r.center[0]), int(r.center[1]))
			cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

			# (x,y,z) are the tag's origin position in camera's frame.
			# Camera's frame being fixed (with Z-axis heading to the tag, Y-axis pointing down),
			# we can conclude how the camera moved and accordingly how we should get back to reference.
			z =r.pose_t[2][0]

			if cX - ref_cX < -5:
				deltaX = 1 # Go right. (since the camera is at the back of the car)
			elif cX - ref_cX > 5:
				deltaX = -1 # Go left.
			else:
				deltaX = 0

			if cY - ref_cY < -5:
				deltaY = -1 # Go up.
			elif cY - ref_cY > 5:
				deltaY = 1 # Go down.
			else:
				deltaY = 0

			if z - ref_Z < -0.01:
				deltaZ = -1 # GO front.
			elif z - ref_Z > 0.01:
				deltaZ = 1 # Go back.
			else:
				deltaZ = 0

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
	global ref_Z, z, ref_ptA, ptA, ref_ptB, ptB, ref_ptC, ptC, ref_ptD, ptD, ref_cX, ref_cY, cX, cY

	data = request.form['reset']
	ref_ptA = ptA
	ref_ptB = ptB
	ref_ptC = ptC
	ref_ptD = ptD
	ref_cX  = cX
	ref_cY  = cY
	ref_Z   = z

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
