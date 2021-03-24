#!/usr/bin/python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved. 
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
"""
Can use with ROS 
subscribing compressed image message and
publishing compressed image message
"""
#

import jetson.inference
import jetson.utils
import rospy
import cv2
import numpy as np
import subprocess, shlex, psutil
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError


net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
#camera = jetson.utils.gstCamera(1280,720,"/dev/video0")    # '/dev/video0' for V4L2
compmsg = CompressedImage()
rospy.init_node("visualizer")


def detection_and_publish(data) :
	np_arr = np.fromstring(data.data, np.uint8)
	image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).astype(np.float32)
	img = jetson.utils.cudaFromNumpy(img)
	#while True:
	#img, width, height = camera.CaptureRGBA(zeroCopy=1)
	jetson.utils.cudaDeviceSynchronize()
	detections = net.Detect(img, image.shape[1], image.shape[0])
	for list in detections :
		if list.ClassID == 1 :
			print("find person")
	numpyImg = jetson.utils.cudaToNumpy(img, image.shape[1], image.shape[0], 4)
	aimg1 = cv2.cvtColor(numpyImg.astype(np.uint8), cv2.COLOR_RGB2BGR)
	compmsg.header.stamp = rospy.Time.now()
	compmsg.format = "jpeg"
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
	compmsg.data = np.array(cv2.imencode('.jpg', aimg1, encode_param)[1]).tostring()
	comp_img_pub.publish(compmsg)

def start_cam() :
	command ="rosparam set /camera_nano/usb_cam/image_raw/compressed/jpeg_quality 50"
	command = shlex.split(command)
	subprocess.Popen(command)
	command ="roslaunch usb_cam usb_nano_cam.launch"
	command = shlex.split(command)
	subprocess.Popen(command)

start_cam()
rospy.Subscriber('/camera_nano/usb_cam/image_raw/compressed', CompressedImage, detection_and_publish)
comp_img_pub = rospy.Publisher("/camera_nano/object_detect/image_raw/compressed", CompressedImage, queue_size = 1)
rospy.spin()
#display.RenderOnce(img,width, height)
#display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

