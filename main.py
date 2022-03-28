import math
# import subprocess
import os
import seaborn
import numpy
import numpy as np
import csv
import threading
import time
import cv2
import pyrealsense2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class openfaceThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print("Starting execution thread\n")
        os.system(
            "/home/evan/Desktop/OpenFace/OpenFace/build/bin/./FeatureExtraction -device 6 -of 50cmtrial1.csv "
            "-out_dir /home/evan/PycharmProjects/zTesting/face4ztests")
        # "/home/evan/Desktop/OpenFace/OpenFace/build/bin/./FeatureExtraction -fdir /home/evan/PycharmProjects/zTesting/teamImages "
        # "-out_dir /home/evan/PycharmProjects/zTesting/teamImagesResults")
        print("Exiting execution thread\n")


def makingarray(path):
    # newarray = []
    newarray = [['frame', 'faceid', 'timestamp', 'gaze_0_z', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y',
                 'eye_lmk_Z_20',
                 'eye_lmk_Z_21', 'eye_lmk_Z_22', 'eye_lmk_Z_23', 'eye_lmk_Z_24', 'eye_lmk_Z_25', 'eye_lmk_Z_26',
                 'eye_lmk_Z_27', 'eye_lmk_Z_48', 'eye_lmk_Z_49', 'eye_lmk_Z_50', 'eye_lmk_Z_51', 'eye_lmk_Z_52',
                 'eye_lmk_Z_53', 'eye_lmk_Z_54', 'eye_lmk_Z_55']]
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_contents = [_ for _ in csv_reader]
        #        face0array = []
        #        face1array = []
        #        face2array = []
        #        face3array = []
        for line in csv_contents[1:-1]:
            newarray.append(
                [int(line[0]), int(line[1]), float(line[2]), float(line[7]), float(line[10]), float(line[11]),
                 float(line[12]), float(line[257]),
                 float(line[258]), float(line[259]), float(line[260]), float(line[261]), float(line[262]),
                 float(line[263]), float(line[264]), float(line[285]), float(line[286]), float(line[287]),
                 float(line[288]), float(line[289]), float(line[290]), float(line[291]), float(line[292]),
                 float(line[607]), float(line[608]), float(line[609]), float(line[610]), float(line[611]),
                 float(line[612]), float(line[614]), float(line[615]), float(line[616]), float(line[617]),
                 float(line[618]), float(line[619])])
    return newarray


def calculatezeyelandmark(newarray):
    leftavg = 0
    rightavg = 0
    sumleft = 0
    sumright = 0
    for line in newarray[1:]:
        if line[3] != 0:
            # for line in newarray:
            leftavg = (line[7] + line[8] + line[9] + line[10] + line[11] + line[12] + line[13] + line[14]) / 80
            sumleft = sumleft + leftavg
            # leftz = leftavg/line[3]
            rightavg = (line[15] + line[16] + line[17] + line[18] + line[19] + line[20] + line[21] + line[22]) / 80
            sumright = sumright + rightavg
            # rightz = rightavg/line[4]
            # print("left eye z is: ", leftz)
            # print("right eye z is ", rightz)
    bothaverage = (sumleft + sumright) / (2 * len(newarray))
    print("Average Z of both eyes, using eye landmarks, is ", bothaverage)


def calculatezfacelandmark(newarray):
    leftavg = 0
    rightavg = 0
    sumleft = 0
    sumright = 0
    for line in newarray[1:]:
        if line[3] != 0:
            # for line in newarray:
            leftavg = (line[23] + line[24] + line[25] + line[26] + line[27] + line[28]) / 60
            sumleft = sumleft + leftavg
            # leftz = leftavg/line[3]
            rightavg = (line[29] + line[30] + line[31] + line[32] + line[33] + line[34]) / 60
            sumright = sumright + rightavg
            # rightz = rightavg/line[4]
            # print("left eye z is: ", leftz)
            # print("right eye z is ", rightz)
    bothaverage = (sumleft + sumright) / (2 * len(newarray))
    print("Average Z of both eyes, using face landmarks, is ", bothaverage)


# openface_thread = openfaceThread().start()
# time.sleep(12)
# os.system("pkill FeatureExtracti")
twentyfivearray = makingarray('/home/evan/PycharmProjects/zTesting/face4ztests/25cmtrial0.csv')
thirtycmarray = makingarray('/home/evan/PycharmProjects/zTesting/face4ztests/30cmtrial0.csv')
fourtycmarray = makingarray('/home/evan/PycharmProjects/zTesting/face4ztests/40cmtrial0.csv')
fiftycmarray = makingarray('/home/evan/PycharmProjects/zTesting/face4ztests/50cmtrial0.csv')
calculatezeyelandmark(twentyfivearray)
calculatezfacelandmark(twentyfivearray)
print("Desired output for above is 25 cm")
calculatezeyelandmark(thirtycmarray)
calculatezfacelandmark(thirtycmarray)
print("Desired output for above is 30 cm")
calculatezeyelandmark(fourtycmarray)
calculatezfacelandmark(fourtycmarray)
print("Desired output for above is 40 cm")
calculatezeyelandmark(fiftycmarray)
calculatezfacelandmark(fiftycmarray)
print("Desired output for above is 50 cm")
