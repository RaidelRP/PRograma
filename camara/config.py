

HAAR_FACES         = 'haarcascade_frontalface_default.xml'
HAAR_SCALE_FACTOR  = 1.2
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (20, 20)



DEBUG_IMAGE = 'capture.jpg'

def get_camera():
        import picam
        return picam.OpenCVCapture()

