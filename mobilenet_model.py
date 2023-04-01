import tensorflow as tf
from config import config
from util import bilinear_sampler

#import all necessary layers
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, Lambda, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model

def refine(base,offsets,num_steps=2):
    for i in range(num_steps):
        base = base + bilinear_sampler(offsets,base)
    return base

def split_and_refine_mid_offsets(args):
    mid_offsets, short_offsets = args
    output_mid_offsets = []
    for mid_idx, edge in enumerate(config.EDGES+[edge[::-1] for edge in config.EDGES]):
        to_keypoint = edge[1]
        kp_short_offsets = tf.stack([short_offsets[:,:,:,to_keypoint],
                                     short_offsets[:,:,:,config.NUM_KP + to_keypoint]], axis=3)
        kp_mid_offsets = mid_offsets[:,:,:,2*mid_idx:2*mid_idx+2]
        kp_mid_offsets = refine(kp_mid_offsets,kp_short_offsets,2)
        output_mid_offsets.append(kp_mid_offsets)
    return tf.concat(output_mid_offsets,axis=-1)

def mobilenet_block (x, filters, strides):
    
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def model(isTraining = True, is_person_detection= False):
    input = Input(shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],3))
    x = tf.pad(input,tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]]))
    
    x = Conv2D(filters = 24, kernel_size = 3, strides = 2, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # main part of the model
    x = mobilenet_block(x, filters = 48, strides = 1)
    x = mobilenet_block(x, filters = 96, strides = 2)
    x = mobilenet_block(x, filters = 96, strides = 2)
    x = mobilenet_block(x, filters = 192, strides = 1)
    x = mobilenet_block(x, filters = 192, strides = 1)
    x = mobilenet_block(x, filters = 384, strides = 2)
    for i in range(7):
        x = mobilenet_block(x, filters = 384, strides = 1)
    
    if is_person_detection:
        x =  Conv2D(filters = 1280, kernel_size = 1, strides = 1, padding = 'same')(x)
        classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid")])
        outputs = classifier(x)
        return Model(inputs=input, outputs=outputs)
    
    mid_offsets =  Conv2D(filters = 4 * config.NUM_EDGES, kernel_size = 1, strides = 1, padding = 'same', name="midoffsets")(x)
    shortoffsets =  Conv2D(filters = 2 * config.NUM_KP, kernel_size = 1, strides = 1, padding = 'same', name="shortoffsets")(x)
    heatmap =  Conv2D(filters = config.NUM_KP, kernel_size = 1, strides = 1, padding = 'same', activation=tf.nn.sigmoid, name="heatmap")(x)
    
    heatmap_lambda = Lambda(lambda x: tf.image.resize(x, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), 
                            method = tf.image.ResizeMethod.BILINEAR), name="heatmap_lambda")(heatmap)
    
    shortoffsets_lambda = Lambda(lambda x: tf.image.resize(x, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), 
                                  method = tf.image.ResizeMethod.BILINEAR), name = "shortoffsets_lambda")(shortoffsets)
    
    mid_offsets_lambda = Lambda(lambda x: tf.image.resize(x, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), 
                                  method = tf.image.ResizeMethod.BILINEAR), name="mid_offsets_lambda")(mid_offsets)

    
    mid_offsets_refined  = Lambda(split_and_refine_mid_offsets, name = "mid_offsets_refined")([mid_offsets_lambda, shortoffsets_lambda])
    if isTraining is True:
        return Model(inputs=input, outputs=[heatmap_lambda, shortoffsets_lambda, mid_offsets_refined, heatmap, shortoffsets, mid_offsets])
    else:
        return Model(inputs=input, outputs=[heatmap, shortoffsets, mid_offsets])