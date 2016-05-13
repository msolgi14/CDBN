# Derived from a test case by Chris Heuser
# Also see FAQ about PyCUDA and threads.


import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import threading
import numpy

import main

class MainThread(threading.Thread):
    def __init__(self, network):
        threading.Thread.__init__(self)
        self.network = network

    def run(self):
        main.main(self.network)
        return

if __name__ == "__main__":
    #main.main(False)
    cuda.init()
    the_thread = MainThread(False)
    the_thread.start()
    
