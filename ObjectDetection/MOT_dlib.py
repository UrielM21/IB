########################################################
########################################################
########            Ing. Muñoz Uriel            ########
########    E-mail: uriel.a.munnoz@gmail.com    ########
########################################################
########################################################

# import the necessary packages
import multiprocessing
from threading import Thread, active_count
import numpy as np
import time
import imutils
from utils import VideoStream, DynamicWriter
from trackers import Tracker_OpenCV, Tracker_Dlib, OPENCV_OBJECT_TRACKERS
import cv2
from imutils.video import FPS
from absl import app, flags, logging
from enum import Enum

class MODE(Enum):
    IMAGE = 0
    VIDEO = 1
    LIVE  = 2

FLAGS = flags.FLAGS
flags.DEFINE_bool("screen", default=True, help="Whether or not to prompt the image to the screen")
flags.DEFINE_bool("use_gpu", default=False, help="Whether or not to use CUDA")
flags.DEFINE_string('model', default=None, help="Model path")
flags.DEFINE_string('config', default=None, help="Config path")
flags.DEFINE_string('labels', default=None, help="Labels textfile")
flags.DEFINE_string('input', default=None, help="Input data for various mode")
flags.DEFINE_string('output', default=None, help="path to optional output video file")
flags.DEFINE_float('update_freq', default=3.0, help="Object detection frequency")
flags.DEFINE_multi_integer('input_size',
                           default=(380, 380),
                           lower_bound=0,
                           help="Input size")
flags.DEFINE_string('log_directory', default=None, help="Log directory")
flags.DEFINE_enum_class(
    'mode',
    default=MODE.VIDEO,
    enum_class=MODE,
    help=
    "Select exec mode, One of {'IMAGE','VIDEO','LIVE'}"
)
flags.DEFINE_float('confidence', default=0.2, help="Minimum probability to filter weak detections")
POISON_PILL = "STOP"

def log(msg):
    """Log message as info"""
    logging.info(msg)

class MOT: # pylint: disable=too-many-instance-attributes
    """Tracker Class"""
    def __init__(self, # pylint: disable=too-many-arguments
                 object_detection_model=None,
                 tracker=None,
                 confidence=0.2,
                 max_staleness=12,
                 labels=None,
                 update_freq=None,
                 screen=True):

        self.status = 'unconfirmed'
        log("Initializing Tracker")

        self.steps_alive = 1
        self.staleness = 0.0
        self.max_staleness = max_staleness
        self.od_model = object_detection_model
        self.tracker = tracker
        self.update_freq = update_freq
        self.update_detection = True
        self.labels = labels
        self.confidence = confidence

        self.input_queues = []
        self.output_queues = []

        self.manager = multiprocessing.Manager()
        self.num_objects_being_tracked = 0
        self.threads = []

        self.screen = screen
        self.lost = 0

        self.fps = FPS().start()
        self.start = time.perf_counter()

    def start_tracking(self, input_stream, output_file = None, track_class = None):
        """Start tracking"""
        log("Starting Tracker")
        # vs = VideoStream(input_stream)
        vs = cv2.VideoCapture(input_stream)
        # vs.start()
        frame_id = 0
        fps = -1
        writer = None
        width = 600
        height = None

        _, original_frame = vs.read()
        # frame = vs.read()

        # Ratios
        (h, w) = original_frame.shape[:2]
        r_w = (float(w) / width) if width is not None else (float(h) / height)
        r_h = (float(h) / height) if height is not None else (float(w) / width)

        frame = imutils.resize(original_frame, width=width)
        if output_file is not None:
            writer = DynamicWriter(output_file, "MJPG", 60.0, (frame.shape[1], frame.shape[0]))
            # fourcc = cv2.VideoWriter_fourcc(*"MJPG") #.avi
            # writer = cv2.VideoWriter(output_file, fourcc, 60.0,
            #                          (frame.shape[1], frame.shape[0]))
        
        start_time = time.perf_counter()
        
        while True: # loop over frames from the video file stream
            # grab the next frame from the video file
            # frame = vs.read()
            _, original_frame = vs.read()
            # check to see if we have reached the end of the video file
            if original_frame is None:
                break
            frame_id += 1

            frame = imutils.resize(original_frame, width=600)
            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # update the FPS counter
            if (frame_id % 20) == 0:
                end_time = time.perf_counter()
                elapsed = (end_time - start_time)
                fps = 20 / elapsed
                # print(f"FPS Aprox.: {fps:.2f}")
                start_time = time.perf_counter()
                
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3, 8)

            if self.update_detection:
                self.status = "Detecting"
                log("Detecting objects")
                self.update_detection = False
                classes, _, boxes = self.detect(frame, track_class=track_class)
                if classes is None:
                    self.lost += 1
                    if self.lost == 3:
                        for iq in self.input_queues:
                            iq.put(POISON_PILL)
                    continue
                self.lost = 0


                for iq in self.input_queues:
                    iq.put(POISON_PILL)
                for th in self.threads:
                    th.join()
                # print(active_count())
                self.num_objects_being_tracked = 0
                self.threads = []
                self.input_queues = []
                self.output_queues = []
                for box, label in zip(boxes, classes):
                    self.num_objects_being_tracked += 1
                    
                    iq = self.manager.Queue()
                    oq = self.manager.Queue()
                    self.input_queues.append(iq)
                    self.output_queues.append(oq)
                    p = multiprocessing.Process(target = self._start_tracker,
                                                args = (box, label, frame, iq, oq),
                                                daemon=True)
                    # p = Thread(target = self._start_tracker,
                    #             args = (box, label, frame, iq, oq),
                    #             daemon=True)
                    p.start()
                    self.threads.append(p)

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (box[0], box[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                
                # cv2.imshow("Frame", frame)
                # cv2.waitKey(-1)
                
                self.start = time.perf_counter()

            else:
                self.status = "Tracking"
                # log("Tracking object")
                for iq in self.input_queues:
                    iq.put(frame)

                for oq in self.output_queues:
                    if oq.empty():
                        continue
                    (label, (startX, startY, endX, endY)) = oq.get()
                    # startX = int(startX * r_w)
                    # startY = int(startY * r_h)
                    # endX = int(endX * r_w)
                    # endY = int(endY * r_h)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            self.fps.update()
            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)
                # writer.write(original_frame)
            # show the output frame
            if self.screen:
                cv2.imshow("Frame", frame)
            self.steps_alive += 1
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            
            if self.update_freq and (time.perf_counter() - self.start) > self.update_freq:
                self.start = time.perf_counter()
                self.update_detection = True

        self.status = "Tracker stopped"
        log("Tracker stopped")
        # stop the timer and display FPS information
        self.fps.stop()
        log("elapsed time: {:.2f}".format(self.fps.elapsed()))
        log("approx. FPS: {:.2f}".format(self.fps.fps()))
        # check to see if we need to release the video writer pointer
        if writer is not None:
            writer.release()


        # do a bit of cleanup
        for iq in self.input_queues:
            iq.put(POISON_PILL)
        for th in self.threads:
            th.join()
        cv2.destroyAllWindows()
        # vs.stop()
        vs.release()

    def detect(self, frame, track_class=None):
        """Use object detection model to update boxes"""
        classes, confidences, boxes = self.od_model.detect(frame,
                                                        confThreshold=self.confidence,
                                                        nmsThreshold=0.)
        classes = np.array(classes).flatten()
        
        if self.num_elements(classes) == 0:
            return None, None, None
        if self.labels:
            if track_class:
                classes = [self.labels[x-1] for x in classes if self.labels[x-1] in track_class]
            else:
                classes = [self.labels[x-1] for x in classes]
        else:
            classes = [str(x-1) for x in classes]
        print(classes)
        # confidences_idx = [idx for idx, conf in enumerate(confidences) if conf > self.confidence]
        # (left, top, width, height) = box
        # (startX, startY, endX, endY) = (left, top, left + width - 1, top + height - 1)
        # boxes[:, 0] = boxes[:, 0]
        # boxes[:, 1] = boxes[:, 1]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - 1
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - 1
        return classes, confidences, boxes

    def num_elements(self, array):
        return array.ndim and array.size

    def _start_tracker(self, box, label, frame, iq, oq):
        """Start tracker process"""
        self.tracker.init(frame, box)

        while True:
            if iq.empty():
                continue
            frame = iq.get()
            if isinstance(frame, str) and frame == POISON_PILL:
                break

            success, (startX, startY, endX, endY) = self.tracker.update(frame)
            if success:
                oq.put((label, (startX, startY, endX, endY)))
            else:
                print("No success")
                break



def main(argv = None):
    """ main """
    net = cv2.dnn_DetectionModel("ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb",
                                "ssd_mobilenet_v3_large_coco_2020_01_14/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    if FLAGS.use_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    with open("labels_coco.txt", 'r') as f:
        labels = f.read().splitlines()

    # tracker_obj = Tracker_OpenCV(OPENCV_OBJECT_TRACKERS["kcf"])
    tracker_obj = Tracker_Dlib()
    tracker = MOT(object_detection_model=net,
                  tracker=tracker_obj,
                  confidence=0.5,
                  max_staleness=12,
                  labels=labels,
                  update_freq=FLAGS.update_freq,
                  screen=FLAGS.screen,)

    tracker.start_tracking("person_walking.mp4", output_file=FLAGS.output, track_class=['person'])
    # tracker.start_tracking("Couple holding hands -H264 75.mov", output_file=FLAGS.output, track_class=['person'])

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    logging.set_verbosity(logging.INFO)
    app.run(main)