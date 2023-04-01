import tensorflow as tf   
import  tensorflow.keras.applications.resnet_v2 as resnet_v2
from config import config
from util import bilinear_sampler

from tensorflow.keras.layers import Input, Conv2D, Lambda
from tensorflow.keras import Model

def refine(base,offsets,num_steps=2):
    for i in range(num_steps):
        base = base + bilinear_sampler(offsets, base)
    return base

def split_and_refine_mid_offsets(args):
    mid_offsets, short_offsets = args
    output_mid_offsets = []
    for mid_idx, edge in enumerate(config.EDGES+[edge[::-1] for edge in config.EDGES]):
        to_keypoint = edge[1]
        kp_short_offsets = tf.stack([short_offsets[:,:,:,to_keypoint],
                                     short_offsets[:,:,:,config.NUM_KP+to_keypoint]], axis=3)
        kp_mid_offsets = mid_offsets[:,:,:,2*mid_idx:2*mid_idx+2]
        kp_mid_offsets = refine(kp_mid_offsets,kp_short_offsets,2)
        output_mid_offsets.append(kp_mid_offsets)
    return tf.concat(output_mid_offsets,axis=-1)


def model(isTraining = True):
    input = Input(shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],3))
    Resnet = resnet_v2.ResNet101V2(include_top=False,weights='imagenet', 
                            input_shape=([config.IMAGE_SHAPE[0], 
                                          config.IMAGE_SHAPE[1],
                                          3]))
    # print(net)
    mid_offsets =  Conv2D(filters = 4 * config.NUM_EDGES, kernel_size = 1, 
                          strides = 1, padding = 'same', name="midoffsets")(Resnet.output)
    shortoffsets =  Conv2D(filters = 2 * config.NUM_KP, kernel_size = 1, strides = 1, padding = 'same', name="shortoffsets")(Resnet.output)
    heatmap =  Conv2D(filters = config.NUM_KP, kernel_size = 1, strides = 1, padding = 'same', activation=tf.nn.sigmoid, name="heatmap")(Resnet.output)
    
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

