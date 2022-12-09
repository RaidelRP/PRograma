import io
import time
import cv2
import numpy as np
import picamera
import config


class OpenCVCapture(object):
  def read(self):
    data = io.BytesIO()
    
    # Resolucion original
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    #CAMERA_WIDTH = 1280
    #CAMERA_HEIGHT = 720
    
    with picamera.PiCamera() as camera:
      camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)  
      camera.rotation = (-90)
      camera.capture(data, format='jpeg')
    data = np.fromstring(data.getvalue(), dtype=np.uint8)

    image = cv2.imdecode(data, 1)

    return image
