from keras import backend as K
from keras.models import *
from keras.layers import *

import os
import cv2
import numpy as np
import itertools
import random
import colorsys

# Why not?
class doubleIteratorUp:
    def __init__(self, iteration):
        self.iteration = iteration
        self.num = 1
    def __iter__(self):
        return self
    def __next__(self):
        if self.iteration != 0:
            last = self.num
            self.num = self.num * 2
            self.iteration -= 1
            return last
        raise StopIteration


class doubleIteratorDown:
    def __init__(self, iteration):
        self.iteration = iteration
        self.num = 2 ** (iteration-1)
    def __iter__(self):
        return self
    def __next__(self):
        if self.iteration != 0:
            last = self.num
            self.num = self.num // 2
            self.iteration -= 1
            return last
        raise StopIteration

# Convolution layer, 
# need to think about other normalization methods 
# instead of batch normalization
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, activation='relu'):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    return x

# Model of neural network       
def unet_custom(n_classes, image_height, image_width, image_channels=3, model_depth=3,
              activation='relu', n_filters=32, dropout=0.2, batchnorm = True):

    list_conv = []
    list_layers = []
    input_layer = Input(shape=(image_height, image_width, image_channels))

    list_layers.append(input_layer)
    
    for i in doubleIteratorUp(model_depth):
        list_conv.append(conv2d_block(list_layers[-1], n_filters * i, kernel_size = 3,\
            batchnorm = batchnorm, activation = activation))
        pool = MaxPooling2D((2, 2))(list_conv[-1])
        list_layers.append(Dropout(dropout)(pool))

    layer_conv_num = len(list_conv) - 1 

    list_layers.append(conv2d_block(list_layers[-1], n_filters * (2**model_depth), kernel_size = 3,\
            batchnorm = batchnorm, activation = activation))

    for i in doubleIteratorDown(model_depth):
        conv2DT = Conv2DTranspose(n_filters * i, (3, 3), strides = (2, 2), padding = 'same')(list_layers[-1])
        concat = concatenate([conv2DT, list_conv[layer_conv_num]])
        layer_conv_num -= 1
        drop = Dropout(dropout)(concat)
        list_layers.append(conv2d_block(drop, n_filters * i, kernel_size = 3,\
            batchnorm = batchnorm, activation = activation))
    
    output_layer = Conv2D(n_classes, (1, 1), activation='sigmoid')(list_layers[-1])

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Loop through all files and record their unique names
def get_file_names(images_path,segs_path):
    image_files = []
    segs_files = []
    for entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, entry)) \
        and os.path.splitext(entry)[1] == '.png':
            image_files.append(entry)

    for entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(images_path, entry)) \
        and os.path.splitext(entry)[1] == '.png':
            if entry in image_files:
                segs_files.append(entry)
    if len(segs_files) != len(image_files):
        raise RuntimeError(f"The length of the list of pictures "
                            f"does not match the length of the list of masks: "
                            f"{len(image_files)}  !=  {len(segs_files)}")
    return image_files

# Norm to [0,1]
def norm_image(image): 
    return (image - np.min(image))/np.ptp(image)

# Function for dividing a dataset into training and validation
# returns a looped iterator
def fetch_delimiter(images_path, segs_path, val_part = 0.05):
    list_names = get_file_names(images_path, segs_path)
    random.shuffle(list_names)

    list_names_train = list_names[int(len(list_names) * val_part) : len(list_names)]
    image_names_train_it = itertools.cycle(list_names_train)

    list_names_val = list_names[0 : int(len(list_names) * val_part)]
    image_names_val_it = itertools.cycle(list_names_val)
    return image_names_train_it, image_names_val_it

# Generator for training and/or validation model
def fetch_gen(image_names_it, images_path, segs_path, batch_size,
                n_classes, height, width):
    
    while True:
        arr_image = []
        arr_seg = []
        for _ in range(batch_size):
            image_name = next(image_names_it)

            image = cv2.imread(images_path + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.float32(cv2.resize(image, (width, height)))

            image = image.astype(np.float32)
            image = norm_image(image)
            arr_image.append(image)

            seg = cv2.imread(segs_path + image_name)

            seg = cv2.resize(seg, (width, height))

            seg_labels = np.zeros((height,width, n_classes))
            seg = seg[:, :, 0]
            for i in range(n_classes):
                seg_labels[:, :, i] = (seg == i).astype(int)
            arr_seg.append(seg_labels)

        yield np.array(arr_image), np.array(arr_seg)

# Model prediction,
# takes the path to the picture
def predict_model(model, path, n_classes, height, width):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = np.float32(cv2.resize(image, (width, height)))
    image = image.astype(np.float32)
    image = norm_image(image)

    pr = model.predict(np.array([image]))[0]
    pr = pr.reshape((height,  width, n_classes)).argmax(axis=2)
    return pr

# Label and picture gluing
def concat_legends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img


# Label picture generator
def get_img_legends(colored_labels,width=100,height=500):

    scale = int(height/len(colored_labels))

    img_legend = np.zeros((height, width, 3),dtype="uint8")
    i = 0
    for key in colored_labels:
        cv2.rectangle(img_legend, (0, (scale*i)),
                                    (width, (scale*(i+1))),
                                    colored_labels[key], -1)
        cv2.putText(img_legend, key, (5, (scale*i)+20), 0, 0.6, (0, 0, 0), 1)
        i+=1

    return img_legend


# def overlay_seg_image(inp_img, seg_img):
#     fused_img = (inp_img/2 + seg_img/2).astype('uint8')
#     return fused_img

# Random color generator for labels
def gen_color_for_labels(labels):
    dict_labels = {}
    for label in labels:
        h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
        dict_labels[label] = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return dict_labels

# Filling the predicted image with color
def get_colored_segmentation_image(seg_arr, colored_labels):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    seg_img = np.zeros((output_height, output_width, 3))
    i = 0
    for c in colored_labels:
        seg_arr_c = seg_arr[:, :] == i
        seg_img[:, :, 0] += ((seg_arr_c)*(colored_labels[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colored_labels[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colored_labels[c][2])).astype('uint8')
        i+=1

    return seg_img

#Combines several functions to display the predicted image in a human-readable form
def visualize_segmentation(seg_arr, path, colored_labels):

    image = cv2.imread(path)

    seg = get_colored_segmentation_image(seg_arr, colored_labels)

    seg = (image/2 + seg/2).astype('uint8')

    img_legend = get_img_legends(colored_labels,height=seg.shape[0])

    seg_img = concat_legends(seg, img_legend)

    return seg_img


def predict_model_visualize(model, path, colored_labels, height, width):

    pr = predict_model(model, path, len(colored_labels), height, width)

    o = visualize_segmentation(pr, path, colored_labels)

    return o

# Metrics
def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

            