import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import tqdm
import imageio.v2 as imageio
import threading
import queue
import numpy as np

class MovieGenerator:

    def __init__(self,
                 temp_dir_path="~/Downloads/odometry_temp") -> None:
        
        self.temp_dir_path = temp_dir_path
        self.temp_file_name = "frame"

        self.next_frame:int = 0

        self.figure:Figure = None
        self.axs:list[Axes] = []

        self.video_file_name = None
        self.fps = 20
        self.writer = None
        self.frame_queue = None
        self.writer_thread = None
        self._stop_event = threading.Event()

        self.reset()

    ####################################################################
    #Helper functions
    #################################################################### 

    def _create_temp_dir(self):

        path = self.temp_dir_path
        if os.path.isdir(path):

            print("found temp dir: {}".format(path))

            # clear the temp directory
            self._clear_temp_dir()

        else:
            print("creating temp directory: {}".format(path))
            os.makedirs(path)

        return

    def _clear_temp_dir(self):

        path = self.temp_dir_path

        if os.path.isdir(path):
            print("clearing temp directory {}".format(path))
            for file in os.listdir(path):

                file_path = os.path.join(path, file)

                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print("Failed to delete {}".format(file_path))

        else:
            print("temp directory {} not found".format(path))

    def _delete_temp_dir(self):

        path = self.temp_dir_path

        if os.path.isdir(path):

            print("deleting temp dir: {}".format(path))

            # clear the directory first
            self._clear_temp_dir()

            # delete the directory
            os.rmdir(path)

        else:
            print("temp directory {} not found".format(path))
    
    ####################################################################
    #Helper functions
    #################################################################### 

    def reset(self):

        self._create_temp_dir()
        self.next_frame = 0

    def initialize_figure(self,nrows,ncols,figsize,wspace=0.3,hspace=0.3):

        self.figure,self.axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize
        )

        self.figure.subplots_adjust(wspace=wspace,hspace=hspace)

    def _writer_worker(self):
        """Background thread that pops frames from the queue and writes them."""
        while not self._stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if self.writer is not None:
                    self.writer.append_data(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue

    def start_movie(self, video_file_name:str="result.mp4", fps:int=20):
        """Starts the asynchronous video writer."""
        self.video_file_name = video_file_name
        self.fps = fps
        self.writer = imageio.get_writer(video_file_name, fps=fps)
        self.frame_queue = queue.Queue(maxsize=100)
        self._stop_event.clear()
        
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def clear_axes(self):

        for ax in self.axs.flat:
            ax.cla()
    
    def save_frame(self,clear_axs = True):
        
        if self.writer is None:
            raise RuntimeError("MovieGenerator.start_movie() must be called before save_frame()")

        # Force a draw so the renderer buffer is updated
        self.figure.canvas.draw()

        # Extract RGB buffer directly from matplotlib
        w, h = self.figure.canvas.get_width_height()
        buf = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)

        # Enqueue the buffer for the background thread
        try:
            self.frame_queue.put(buf.copy(), timeout=2.0)
        except queue.Full:
            print("Warning: MovieGenerator queue is full, dropping frame.")

        self.next_frame+=1

        #clear the axes if desired
        if clear_axs:
            self.clear_axes()

    
    def save_movie(self):
        """Closes the background thread and finalizes the video file."""
        if self.writer_thread is not None:
            self._stop_event.set()
            self.writer_thread.join()
            
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            
        print(f"Movie saved successfully with {self.next_frame} frames.")
    