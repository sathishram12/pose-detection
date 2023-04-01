
class config:

    #########
    # POSE CONFIGS:
    #########

    # Number of keypoints
    NUM_KP = 17

    # List of keypoint names
    KEYPOINTS = [
        "nose",         # 0
        # "neck",
        "Leye",         # 1
        "Reye",         # 2
        "Lear",         # 3
        "Rear",         # 4
        "Lshoulder",    # 5
        "Rshoulder",    # 6
        "Lelbow",       # 7
        "Relbow",       # 8
        "Lwrist",       # 9
        "Rwrist",       # 10
        "Lhip",         # 11
        "Rhip",         # 12
        "Lknee",        # 13
        "Rknee",        # 14
        "Lankle",       # 15
        "Rankle",       # 16
      
    ]

    # Indices of right and left keypoints (for flipping in augmentation)
    RIGHT_KP = [2, 4, 6, 8, 10, 12, 14, 16]
    LEFT_KP =  [1, 3, 5, 7,  9, 11, 13, 15]

    # List of edges as tuples of indices into the KEYPOINTS array
    # (Each edge will be used twice in the mid-range offsets; once in each direction)
    EDGES = [
        ( 0,  1),
        ( 1,  3),
        ( 0,  2),
        ( 2,  4),
        ( 0,  5),
        ( 5,  7),
        ( 7,  9),
        ( 5, 11),
        (11, 13),
        (13, 15),
        ( 0,  6),
        ( 6,  8),
        ( 8, 10),
        ( 6, 12),
        (12, 14),
        (14, 16)
    ]

    NUM_EDGES = len(EDGES)

    #########
    # PRE- and POST-PROCESSING CONFIGS:
    #########

    # Radius of the discs around the keypoints. Used for computing the ground truth
    # and computing the losses. (Recommended to be a multiple of the output stride.)
    KP_RADIUS = 16

    # The threshold for extracting keypoints from hough maps.
    PEAK_THRESH = 0.001

    #########
    # TRAINING CONFIGS:
    #########

    # Input shape for training images (By convention s*n+1 for some integer n and s=output_stride)
    IMAGE_SHAPE = (353, 481, 3)

    # Output stride of the base network (resnet101 or resnet152 in the paper)
    # [Any convolutional stride in the original network which would reduce the 
    # output stride further is replaced with a corresponding dilation rate.]
    OUTPUT_STRIDE = 16

    # Weights for the losses applied to the keypoint maps ('heatmap'), the binary segmentation map ('seg'),
    # and the short-, mid-, and long-range offsets.
    '''
    LOSS_WEIGHTS = {
        'heatmap': 4,
        'seg': 2,
        'short': 1,
        'mid': 0.25,
        'long': 0.125
    }
    '''
    LOSS_WEIGHTS = {
        'heatmap': 4,
        'mid': 1,
        'short': 2,
    }
    # Batch_size
    BATCH_SIZE = 1

    # Learning Rate
    LEARNING_RATE = 4e-5

    # Whether to keep the batchnorm weights frozen.
    BATCH_NORM_FROZEN = True

    # Number of GPUs to distribute across
    NUM_GPUS = 1

    # The total batch size will be (NUM_GPUS * BATCH_SIZE_PER_GPU)
    BATCH_SIZE_PER_GPU = 1

    # Whether to use Polyak weight averaging as mentioned in the paper
    POLYAK = False

    # Optional model weights filepath to use as initialization for the weights
    LOAD_MODEL_PATH = None

    # Where to save the model.
    SAVE_MODEL_PATH = './model/personlab/'

    # Where to save the pretrained model
    PRETRAINED_MODEL_PATH = './model/101/resnet_v2_101.ckpt'

    # Epochs
    MAX_EPOCHS = 20
    
    # Where to save the coco2017 dataset
    TRAIN_ANNO_FILE = '../data/coco2017/annotations/person_keypoints_train2017.json'
    TRAIN_IMG_DIR = '../data/coco2017/train2017'
    
    VAL_ANNO_FILE = '../data/coco2017/annotations/person_keypoints_val2017.json'
    VAL_IMG_DIR = '../data/coco2017/val2017'
    
    # log dir
    LOG_DIR = './log'

class TransformationParams:

    target_dist = 0.8
    scale_prob = 1.
    scale_min = 0.8
    scale_max = 2.0
    max_rotate_degree = 30.
    center_perterb_max = 20.0
    flip_prob = 0.5
