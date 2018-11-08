import subprocess
import os
import cv2
from shutil import copyfile
import numpy as np

"""
This script is meant to pre-process films so that they all have a framerate
and other characteristics that are the same as those of the cartoons
"""

def display_fps(films):
    """Display the fps of each file in a list of files"""
    for i, file in enumerate(films):
        video = cv2.VideoCapture(file)
        fps = video.get(cv2.CAP_PROP_FPS)
    
        print(i, fps, file)

    return 

def convert_mp4s_to_specified_fps(films, film_names, new_dir, new_fps=30, tol=0.05):
    """Convert a list of mp4 files to other mp4s with a specified framerate"""

    for film, name in zip(films, film_names):
        video = cv2.VideoCapture(film)
        fps = video.get(cv2.CAP_PROP_FPS)

        if np.abs(fps-new_fps) < tol:
            copyfile(film, new_dir+name)
        else:
            pass
            #c = "cd C:\Program Files\ffmpeg\ffmpeg-20181106-d96ae9d-win64-static\bin".split(r' ')
            #subprocess.call(c, shell=True, executable="C:\Windows\System32\cmd.exe")
            #c = f'ffmpeg -i {film} -r {new_fps} {new_dir+name}'.split(r' ')
            #subprocess.call(c, shell=True, executable="C:\Windows\System32\cmd.exe")
    return

if __name__ == "__main__":

    film_dir = "H:/video-to-video/data/film/"
    film_names = os.listdir(film_dir)
    films = [film_dir + f for f in film_names]

    display_fps(films)

    new_dir = "H:/video-to-video/data/film-30fps/"
    convert_mp4s_to_specified_fps(films, film_names, new_dir, new_fps=30, tol=0.05)