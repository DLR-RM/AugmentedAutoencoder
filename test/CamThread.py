import numpy as np
import cv2
import pygame
import pygame.camera
import time
import threading
from threading import Thread, Event, ThreadError

class CamThread():

    def __init__(self, width, height):

        self.thread_cancelled = False
        self.thread = Thread(target=self.run)
        self.thread.daemon = True                            # Daemonize thread
        self.img = None
        self.width = width
        self.height = height
        self.cam = False
        print "camera initialised"
        self.random = np.random.rand()


    def startThread(self):
        self.thread.start()
        print "camera stream started"

    def run(self):


        pygame.init()
        pygame.camera.init()
        self.cam = pygame.camera.Camera("/dev/video0",(self.width,self.height))
        self.cam.start()

        #setup window
        windowSurfaceObj = pygame.display.set_mode((self.width,self.height),1,16)
        pygame.display.set_caption('Camera')

        while not self.thread_cancelled:
            try:
                if self.cam.query_image():
                    image = self.cam.get_image()
                    arr = pygame.surfarray.array3d(image)
                    im = np.swapaxes(arr,0,1)
                    self.img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                # if cv2.waitKey(1) ==27:
                #     exit(0)
            except ThreadError:
                self.thread_cancelled = True
                print 'what'
            
            # try:
            #     cv2.imshow('st',img)
            #     cv2.waitKey(1)
            # except:
            #     pass

        
        
    def is_running(self):
        return self.thread.isAlive()
      
    def shut_down(self):
        self.thread_cancelled = True
        #block while waiting for thread to terminate
        while self.thread.isAlive():
            time.sleep(1)
        return True
    
# if __name__ == "__main__":
# width = 720/2#480
# height = 1280/2#640
# cam = CamThread(width,height)
# cam.startThread()

# time.sleep(100)

# while True:

# while True:
#     if cam.img is not None:
#         cv2.imshow('stream',cam.img)
#         cv2.waitKey(1)
#     else:
#         print 'no img'