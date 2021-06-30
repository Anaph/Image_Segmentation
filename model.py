from keras.models import *
from keras.layers import *
import os
import cv2
import numpy as np
import itertools

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
    
    output_layer = Conv2D(n_classes, (1, 1), padding = 'same', activation='sigmoid')(list_layers[-1])

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

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


def train_gen(images_path, segs_path, batch_size,
                n_classes, height, width):
    
    list_names = get_file_names(images_path, segs_path)
    image_names_it = itertools.cycle(list_names)


    while True:
        arr_image = []
        arr_seg = []
        for _ in range(batch_size):
            image_name = next(image_names_it)

            image = cv2.imread(images_path + image_name)
            # image = np.float32(cv2.resize(image, (width, height)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            image -= 127.5 #bad norm
            arr_image.append(image)

            seg = cv2.imread(segs_path + image_name)
            # seg = cv2.resize(seg, (width, height))
            seg_labels = np.zeros((height,width, n_classes))
            seg = seg[:, :, 0]
            for i in range(n_classes):
                seg_labels[:, :, i] = (seg == i).astype(int)
            arr_seg.append(seg_labels)

        yield np.array(arr_image), np.array(arr_seg)

def predict_model(model, path, n_classes, height, width):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(cv2.resize(image, (width, height)))

    image -= 127.5 #bad norm

    pr = model.predict(np.array([image]))[0]
    pr = pr.reshape((height,  width, n_classes)).argmax(axis=2)
    return pr



            