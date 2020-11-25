import imutils
import time
import cv2
import dlib

# OpenCV object tracker implementations

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create,
    'goturn': cv2.TrackerGOTURN_create
}



class Tracker_OpenCV:
    def __init__(self, tracker):
        self._tracker = tracker

    def init(self, frame, box):
        self.tracker = self._tracker()
        (startX, startY, endX, endY) = box
        box = (startX, startY, endX - startX, endY - startY)
        # box = tuple(box)
        self.tracker.init(frame, box)

    def update(self, frame):
        (success, box) = self.tracker.update(frame)
        (x, y, w, h) = [int(v) for v in box]
        (startX, startY, endX, endY) = (x, y, x + w, y + h)
        return success, (startX, startY, endX, endY)

class Tracker_Dlib:
    def __init__(self, tracker = dlib.correlation_tracker):
        self._tracker = tracker

    def init(self, frame, box):
        ''' 
        frame: RGB
        box: (startX, startY, endX, endY)
        '''
        self.tracker = self._tracker()
        rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.tracker.start_track(frame, rect)

    def update(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.tracker.update(frame)
        pos = self.tracker.get_position()

        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
        success = True
        return success, (startX, startY, endX, endY)