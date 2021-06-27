import cv2
import numpy as np

from time import time
from motion_detection.detector import MotionDetector
from motion_detection.packer import pack_images
from numba import jit
import pandas as pd
import sys, getopt


@jit(nopython=True)
def filter_fun(b):
    return ((b[2] - b[0]) * (b[3] - b[1])) > 300

def mask(img,polygons):
    mask = np.zeros_like(img)
    for color, polygon in polygons:
        cv2.fillPoly(mask, np.array([polygon], dtype=np.int64), color=color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def get_argv(argv):
    inputvideo = ''
    outputimage = ''
    outputmask = ''
    try:
        opts, args = getopt.getopt(argv,"hi:a:p:m:",["ifile=","afile=","pfile=","mfile="])
    except getopt.GetoptError:
        print('test.py -i <inputvideo> -a <inputannotations> -p <outputimage> -m <outputmask>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('test.py -i <inputvideo> -a <inputannotations> -p <outputimage> -m <outputmask>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputvideo = arg
        elif opt in ("-a", "--afile"):
            inputannotations = arg
        elif opt in ("-p", "--pfile"):
            outputimage = arg
        elif opt in ("-m", "--mfile"):
            outputmask = arg
    print('Input file is "', inputvideo)
    print('Input annotations is "', inputannotations)
    print('Output image is "', outputimage)
    print('Output mask is "', outputmask)
    return inputvideo, inputannotations, outputimage, outputmask

def process(inputvideo, inputannotations, outputimage, outputmask):
    cap = cv2.VideoCapture(inputvideo)

    detector = MotionDetector(bg_history=1,
                              bg_skip_frames=1,
                              movement_frames_history=8,
                              brightness_discard_level=12,
                              bg_subs_scale_percent=0.5,
                              pixel_compression_ratio=0.2,
                              group_boxes=False,
                              expansion_step=1)

    ctr = 0
    columns = ['Track ID', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
    labels = ['Biker', 'Pedestrian', 'Skateboarder', 'Cart', 'Car', 'Bus']
    df = pd.read_csv(inputannotations, delimiter = " ")
    df.columns = columns
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break


        movement = detector.detect_move(frame)

        boxes = df[df.frame==ctr]
        rect = []
        color = []
        for index, row in boxes.iterrows():
            rect.append([[row['xmin'], row['ymax']],[row['xmax'], row['ymax']], [row['xmax'], row['ymin']], [row['xmin'], row['ymin']]])
            for i in range(len(labels)):
                if row['label'] == labels[i]:
                    color.append(i+1)
        rect = zip(color, rect)
        
        ctr += 1
        
        scale_percent = 200 # percent of original size
        width = int(movement.shape[1] * scale_percent / 100)
        height = int(movement.shape[0] * scale_percent / 100)
        dim = (width, height)

        kernel = np.ones((3,3),np.uint8)
        resized = cv2.dilate(movement,kernel,iterations=2)
        # resize image
        resized = cv2.resize(resized, dim, interpolation = cv2.INTER_AREA)
        resized = mask(resized,rect)
        backtorgb = cv2.cvtColor(resized,cv2.COLOR_GRAY2RGB)
        # cv2.imshow('diff_frame', backtorgb)

        cv2.imwrite(outputimage + str(ctr) + '.png', frame)
        cv2.imwrite(outputmask + str(ctr) + '.png', backtorgb)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break




if __name__ == "__main__":
    inputvideo, inputannotations, outputimage, outputmask = get_argv(sys.argv[1:])
    process(inputvideo, inputannotations, outputimage, outputmask)
