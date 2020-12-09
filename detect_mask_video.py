# USAGE
# python detect_mask_video.py

import json
import imutils
import cv2
import Arduino_driver
import Sensors_driver
import os
import random
import argparse
import pygame
import threading

from datetime import datetime
from detector import Detector, FaceDetection, MaskDetection, AgeGenderDetector

config_file_name = "config.json"

parameters = {"ArduinoPort": "COM9",
              "SensorsPort": "COM14",
              "Cam_src": 0}

try:
    with open(config_file_name) as conf_file:
        parameters = json.load(conf_file)
except FileNotFoundError:
    with open(config_file_name, 'w') as conf_file:
        json.dump(parameters, conf_file)
        print("Default parameters loaded")

pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--face", type=str, default="face_detector",
            help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
            help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.6,
            help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--sens", type=float, default=0.85, 
			help="detection sensitivity")
	return vars(ap.parse_args())
	
args = parse_args()

# create reports dir with current date and statistics file
daily_rep_path = datetime.now().strftime("%Y_%m_%d")
path = os.getcwd()
path = os.path.join(path, 'reports', daily_rep_path)
filename = os.path.join(path, 'stats.csv')

if not os.path.exists(path):
    os.makedirs(path)
    with open(filename, 'w') as f:
        f.writelines("Count;Date;Time;Sex;Age\n")

# setup serial port and open
LEDs = Arduino_driver.LED_driver(parameters['ArduinoPort'])
Sensors = Sensors_driver.Sensors_driver(parameters['SensorsPort'])

det = Detector()
det.start_video_stream(parameters['Cam_src'])

face_det = FaceDetection(threshold=args['confidence'], scale=1.05)
mask_det = MaskDetection(threshold=args['sens'])
age_gen_det = AgeGenderDetector()

last_state = True
last_length = 0
counter = 0
mask_detected_prob = -1
mask_detection_sens = args['sens']

while True:
	inference_time = 0
	faults = 0
	new_length = 0
	frame = det.process_video()
	fd_results = face_det.predict(frame)
	face_bboxes = fd_results["process_output"]["bbox_coord"]
	inference_time += fd_results["predict_end_time"]
	if face_bboxes:
		for face_bbox in face_bboxes:
            # extract the face ROI
			(x, y, w, h) = face_bbox
			face = frame[y:h, x:w]
			md_results = mask_det.predict(face, show_bbox=False, frame=frame, gray_enabled=True)
			mask_detected_prob = md_results["process_output"]["mask_no_mask"]
			if mask_detected_prob > mask_detection_sens:
				faults += 1

			age_gender = age_gen_det.predict(face)

			face_det.draw_output(frame, xmin=x, ymin=y, xmax=w, ymax=h, mask_detected=mask_detected_prob, threshold=mask_detection_sens, 
								center_point=False, ethnics = age_gender["process_output"])
			
			inference_time +=  md_results["predict_end_time"]
			inference_time += age_gender["predict_end_time"]

	fps = 1000. / inference_time
	print("inference time = {:0.2f}, fps = {:0.2f}, confidence = {}, faults = {}" .format(inference_time, fps, mask_detected_prob, faults))
	
	new_length = len(face_bboxes)

	frame = imutils.resize(image=frame, width=600)
	cv2.imshow('frame', frame)

	if new_length == last_length:
		if faults > 0 and last_state == True:
			last_state = False
			pygame.mixer.music.play()
			date_time_parsed = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
			img = os.path.join(path, f"file_{str(date_time_parsed)}.jpg")
			cv2.imwrite(img, frame)
			LEDs.send_state(b'2')
		elif faults==0 and new_length > 0 and last_state == False:
			last_state = True
			LEDs.send_state(b'1')
	else:
		last_state = True
		if faults > 0:
			pygame.mixer.music.play()
			date_time_parsed = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
			img = os.path.join(path, f"file_{str(date_time_parsed)}.jpg")
			cv2.imwrite(img, frame)
			LEDs.send_state(b'2')
		elif faults==0 and new_length > 0:
			LEDs.send_state(b'1')
		else:
			LEDs.send_state(b'0')

	# simple ppl counting
	if new_length > last_length:
		counter += new_length - last_length
		with open(filename, 'a') as f:
			date_time_parsed = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
			parsed_date = date_time_parsed[:10]
			parsed_time = date_time_parsed[11:-3]
			parsed_age = random.randint(15, 60)
			f.writelines(f"{new_length - last_length};{parsed_date};{parsed_time};M;{parsed_age}\n")
	
	last_length = new_length
	print(f"People counted: {counter}")

	if (cv2.waitKey(1)) == ord("q"):
		break

del LEDs
del Sensors
