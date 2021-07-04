import cv2
import numpy as np


from motion_detector.detector import MotionDetector
import pandas as pd
import sys, getopt

# Masking function by rectangles
def mask(img,polygons):
    mask = np.zeros_like(img)
    for color, polygon in polygons:
        cv2.fillPoly(mask, np.array([polygon], dtype=np.int64), color=color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

# Just input
def get_argv(argv):
    inputvideo = ''
    inputannotations = ''
    outputimage = ''
    outputmask = ''
    outputheight = 544
    outputwidth = 736
    try:
        opts, args = getopt.getopt(argv,"hi:a:p:m:",
            ["height=","width=","ifile=","afile=","pfile=","mfile="])
    except getopt.GetoptError:
        print('test.py -i <inputvideo> -a <inputannotations> -p <outputimage> -m <outputmask>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('test.py -i <inputvideo> -a <inputannotations> '+
                            '-p <outputimage> -m <outputmask>' + 
                            '--height <outputheight>' +
                            '--width <outputwidth>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputvideo = arg
        elif opt in ("-a", "--afile"):
            inputannotations = arg
        elif opt in ("-p", "--pfile"):
            outputimage = arg
        elif opt in ("-m", "--mfile"):
            outputmask = arg
        elif opt == "--height":
            outputheight = arg
        elif opt == "--width":
            outputwidth = arg
    print('Input file is "', inputvideo)
    print('Input annotations is "', inputannotations)
    print('Output image is "', outputimage)
    print('Output mask is "', outputmask)
    return inputvideo, inputannotations, outputimage, outputmask, \
        int(outputheight), int(outputwidth)


# Main markup function
def process(inputvideo,inputannotations,outputimage,
            outputmask,outputheight,outputwidth):
    cap = cv2.VideoCapture(inputvideo)

    # Motion detector with tailored parameters
    detector = MotionDetector(bg_history=1,
                              bg_skip_frames=1,
                              movement_frames_history=8,
                              brightness_discard_level=11,
                              bg_subs_scale_percent=0.50,
                              pixel_compression_ratio=0.5,
                              group_boxes=False,
                              expansion_step=1)

    # Frame number
    ctr = 0

    # Constant declaration of dataset parameters
    columns = ['Track ID', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
    labels = ['Biker', 'Pedestrian', 'Skateboarder', 'Cart', 'Car', 'Bus']
    df = pd.read_csv(inputannotations, delimiter = " ")
    df.columns = columns
    
    # Main loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break
        
        # Init frame_prev
        if ctr == 0:
            dim = (outputwidth,outputheight)
            frame_prev = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        movement = detector.detect_simple(frame)

        # Catch every fifth frame
        if ctr%5 != 0:
            ctr += 1
            continue

        # Prepare labels for the mask
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
        
        # Use dilate
        kernel = np.ones((3,3),np.uint8)
        dilate = cv2.dilate(movement,kernel,iterations=2)

        # Reshape to original to use masking
        scale_percent = 200 # percent of original size
        width = int(movement.shape[1] * scale_percent / 100)
        height = int(movement.shape[0] * scale_percent / 100)
        dim = (width, height)
        dilate_resized = cv2.resize(dilate, dim, interpolation = cv2.INTER_AREA)

        # Masking using dataset markup
        masked = mask(dilate_resized,rect)
        backtorgb = cv2.cvtColor(masked ,cv2.COLOR_GRAY2RGB)
        dim = (outputwidth,outputheight)

        # Reshape to output
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        backtorgb = cv2.resize(backtorgb, dim, interpolation = cv2.INTER_AREA)

        # cv2.imshow('backtorgb', (backtorgb/2+frame_prev/2).astype("uint8"))
        # cv2.imshow('diff_frame', backtorgb)
        # cv2.imshow('frame', frame)
        
        # Write
        cv2.imwrite(outputimage + str(ctr) + '.png', frame_prev)
        cv2.imwrite(outputmask + str(ctr) + '.png', backtorgb)

        # Using the previous frame to shift the image
        frame_prev = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == "__main__":
    process(*get_argv(sys.argv[1:]))
