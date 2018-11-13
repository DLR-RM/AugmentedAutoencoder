#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:53:52 2017

@author: GustavZ
"""
import datetime
import cv2
import threading
import time
import tensorflow as tf

import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import Queue
elif PY3:
    import queue as Queue


class FPS:
# from https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
    
    
class FPS2:
    def __init__(self, interval):
        self._glob_start = None
        self._glob_end = None
        self._glob_numFrames = 0
        self._local_start = None
        self._local_numFrames = 0
        self._interval = interval
        self.curr_local_elapsed = None
        self.first = False

    def start(self):
        self._glob_start = datetime.datetime.now()
        self._local_start = self._glob_start
        return self

    def stop(self):
        self._glob_end = datetime.datetime.now()

    def update(self):
        self.first = True
        curr_time = datetime.datetime.now()
        self.curr_local_elapsed = (curr_time - self._local_start).total_seconds()
        self._glob_numFrames += 1
        self._local_numFrames += 1
        if self.curr_local_elapsed > self._interval:
          print("> FPS: {}".format(self.fps_local()))
          self._local_numFrames = 0
          self._local_start = curr_time

    def elapsed(self):
        return (self._glob_end - self._glob_start).total_seconds()

    def fps(self):
        return self._glob_numFrames / self.elapsed()
    
    def fps_local(self):
        if self.first:
            return round(self._local_numFrames / self.curr_local_elapsed,1)
        else:
            return 0.0
    
    
class WebcamVideoStream:
# with modifications from https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.frame_counter = 1
        self.width = width
        self.height = height
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        #Debug stream shape
        self.real_width = int(self.stream.get(3))
        self.real_height = int(self.stream.get(4))
        print("> Start video stream with shape: {},{}".format(self.real_width,self.real_height))
    
    def start(self):
        # start the thread to read frames from the video stream
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self.frame_counter += 1

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        
    def isActive(self):
        # check if VideoCapture is still Opened
        return self.stream.isOpened

    def resize(self):
        try:
            self.frame = cv2.resize(self.frame, (self.width, self.height)) 
        except:
            print("> Error resizing video stream")
        


class SessionWorker():
# from https://github.com/naisy/realtime_object_detection/blob/master/lib/session_worker.py
# TensorFlow Session Thread
#
# usage:
# before:
#     results = sess.run([opt1,opt2],feed_dict={input_x:x,input_y:y})
# after:
#     opts = [opt1,opt2]
#     feeds = {input_x:x,input_y:y}
#     woker = SessionWorker("TAG",graph,config)
#     worker.put_sess_queue(opts,feeds)
#     q = worker.get_result_queue()
#     if q is None:
#         continue
#     results = q['results']
#     extras = q['extras']
#
# extras: None or frame image data for draw. GPU detection thread doesn't wait result. Therefore, keep frame image data if you want to draw detection result boxes on image.
#
    def __init__(self,tag,graph,config):
        self.lock = threading.Lock()
        self.sess_queue = Queue.Queue()
        self.result_queue = Queue.Queue()
        self.tag = tag
        t = threading.Thread(target=self.execution,args=(graph,config))
        t.setDaemon(True)
        t.start()
        return

    def execution(self,graph,config):
        self.is_thread_running = True
        try:
            with tf.Session(graph=graph,config=config) as sess:
                while self.is_thread_running:
                        while not self.sess_queue.empty():
                            q = self.sess_queue.get(block=False)
                            opts = q["opts"]
                            feeds= q["feeds"]
                            extras= q["extras"]
                            if feeds is None:
                                results = sess.run(opts)
                            else:
                                results = sess.run(opts,feed_dict=feeds)
                            self.result_queue.put({"results":results,"extras":extras})
                            self.sess_queue.task_done()
                        time.sleep(0.005)
        except:
            import traceback
            traceback.print_exc()
        self.stop()
        return

    def is_sess_empty(self):
        if self.sess_queue.empty():
            return True
        else:
            return False

    def put_sess_queue(self,opts,feeds=None,extras=None):
        self.sess_queue.put({"opts":opts,"feeds":feeds,"extras":extras})
        return

    def is_result_empty(self):
        if self.result_queue.empty():
            return True
        else:
            return False

    def get_result_queue(self):
        result = None
        if not self.result_queue.empty():
            result = self.result_queue.get(block=False)
            self.result_queue.task_done()
        return result

    def stop(self):
        self.is_thread_running=False
        with self.lock:
            while not self.sess_queue.empty():
                q = self.sess_queue.get(block=False)
                self.sess_queue.task_done()
        return
