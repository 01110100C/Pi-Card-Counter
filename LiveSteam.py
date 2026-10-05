import cv2 
import threading 
import time 

class liveStream:

    def __init__(self, width=1280, height=720, framerate=10, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, framerate)

        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False

        ok, frame = self.cap.read()
        if not ok:
            raise Exception("Could not read frame from video device")
        self.frame = frame

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

def _update(self):
    while not self.stopped:
        ok, frame = self.cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        with self.lock:
            self.frame = frame

def get_frame(self):
    with self.lock:
        return self.frame.copy() 
    
def stop(self):
    self.stopped = True
    self.thread.join()
    self.cap.release()