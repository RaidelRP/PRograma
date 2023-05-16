import io
import time
from threading import Thread

import cv2
import numpy as np
import picamera


class OpenCVCapture(Thread):
    def __init__(self):
        Thread.__init__(self)
        # self.image =  np.ndarray([]

    def run(self):
        data = io.BytesIO()

        CAMERA_WIDTH = 640
        CAMERA_HEIGHT = 480

        with picamera.PiCamera() as camera:
            # camera.start_preview()
            camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
            camera.rotation = 180
            #camera.brightness= 65
            #camera.shutter_speed= 300000
            #camera.framerate = 60
            # camera.stop_preview()
            for img in camera.capture_continuous(data, format='jpeg'):
                data.truncate()
                data.seek(0)

                dataimg = np.fromstring(img.getvalue(), dtype=np.uint8)
                self.image = cv2.imdecode(dataimg, 1)

    def get_image(self):
        return self.image
