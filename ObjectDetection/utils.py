# import the necessary packages
import cv2
import time
from threading import Thread
from queue import Queue
import multiprocessing

POISON_PILL = "STOP"
class DynamicWriter:
	''' Dynamic Writer for OpenCV. Writes'''
	def __init__(self, output_file, fourcc, fps, frame_shape):
		self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
		self.output_file = output_file
		self.fps = fps
		self.frame_shape = frame_shape
		self.last_call = None

		self.queue = multiprocessing.Queue()
		self.process = multiprocessing.Process(target=self._write,
								daemon=True)
		self.process.start()


	def write(self, frame):
		# time_stamp in seconds.
		time_stamp = time.perf_counter()
		if self.last_call is None:
			self.last_call = time.perf_counter()
			self.queue.put((frame, 1))
			return

		number_of_frames = max(1, round((time_stamp - self.last_call) * self.fps))
		self.queue.put((frame, number_of_frames))
		self.last_call = time.perf_counter()
		return

	def _write(self):
		writer = cv2.VideoWriter(self.output_file, self.fourcc, self.fps, self.frame_shape)
		while True:
			if self.queue.empty():
				continue
			out = self.queue.get()
			if isinstance(out, str) and out == POISON_PILL:
				break
			(frame, number_of_frames) = out
			for _ in range(number_of_frames):
				writer.write(frame)
		writer.release()


	def release(self):
		self.queue.put(POISON_PILL)
		self.process.join()
		# self.writer.release()

class VideoStream:
	def __init__(self, src, transform=None, queue_size=128, camera=False, name="VideoStream"):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		if transform:
			assert not camera, "Transform can only be applied if camera mode is false"
		self.name = name
		self.stream = cv2.VideoCapture(src)
		self.camera = camera
		if camera:
			(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False
		self.transform = transform

		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queue_size)
		# intialize thread
		self.thread = Thread(target=self.update, name=self.name, args=(), daemon=True)

	def start(self):
		# start a thread to read frames from the file video stream
		self.thread.start()
		return self

	def update(self):
		while self.camera:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				self.stream.release()
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

		# keep looping infinitely
		while not self.camera:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				self.stream.release()
				return

			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stopped = True
					break
					
				if self.transform:
					frame = self.transform(frame)

				# add the frame to the queue
				self.Q.put(frame)
			else:
				time.sleep(0.1)  # Rest for 10ms, we have a full queue

		self.stream.release()



	def read(self):
		# return the frame most recently read
		if self.camera:
			return self.frame

		# return next frame in the queue
		return self.Q.get()

	# Insufficient to have consumer use while(more()) which does
	# not take into account if the producer has reached end of
	# file stream.
	def running(self):
		return self.more() or not self.stopped

	def more(self):
		# return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
		tries = 0
		while self.Q.qsize() == 0 and not self.stopped and tries < 5:
			time.sleep(0.1)
			tries += 1

		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		self.thread.join()