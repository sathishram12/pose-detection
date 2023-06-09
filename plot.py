from matplotlib import pyplot as plt
import matplotlib
import cv2 as cv
import numpy as np
import math
from config import config
from post_proc import get_keypoints

def visualize_short_offsets(offsets, keypoint_id, centers=None, heatmaps=None, radius=config.KP_RADIUS, img=None, every=1,save_path='./'):
    if centers is None and heatmaps is None:
        raise ValueError('either keypoint locations or heatmaps must be provided')
    

    if isinstance(keypoint_id, str):
        if not keypoint_id in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(keypoint_id))
        else:
            keypoint_id = config.KEYPOINTS.index(keypoint_id)
    
    if centers is None:
        kp = get_keypoints(heatmaps)
        kp = [k for k in kp if k['id']==keypoint_id]
        centers = [k['xy'].tolist() for k in kp]

    kp_offsets = offsets[:,:,[keypoint_id, config.NUM_KP+ keypoint_id]]
    masks = np.zeros(offsets.shape[:2]+(len(centers),), dtype='bool')
    idx = np.rollaxis(np.indices(offsets.shape[1::-1]), 0, 3).transpose((1,0,2))
    for j, c in enumerate(centers):
        dists = np.sqrt(np.square(idx-c).sum(axis=-1))
        dists_x = np.abs(idx[:,:,0] - c[0])
        dists_y = np.abs(idx[:,:,1] - c[1])
        masks[:,:,j] = (dists<=radius)
        if every > 1:
            d_mask = np.logical_and(np.mod(dists_x.astype('int32'), every)==0, 
                                    np.mod(dists_y.astype('int32'), every)==0)
            masks[:,:,j] = np.logical_and(masks[:,:,j], d_mask)
    mask = masks.sum(axis=-1) > 0

    I, J = np.nonzero(mask)

    plt.figure()
    if img is not None:
        plt.imshow(img)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 200    
    plt.quiver(J, I, kp_offsets[I,J,0], kp_offsets[I,J,1], color='r', angles='xy', scale_units='xy', scale=1)
    plt.savefig('./demo_result/short_offsets.jpg',bbox_inches = 'tight')
    plt.close()


def visualize_mid_offsets(offsets, from_kp, to_kp, centers=None, heatmaps=None, radius=config.KP_RADIUS, img=None, every=1,save_path='./'):
    if centers is None and heatmaps is None:
        raise ValueError('either keypoint locations or heatmaps must be provided')
    

    if isinstance(from_kp, str):
        if not from_kp in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(from_kp))
        else:
            from_kp = config.KEYPOINTS.index(from_kp)
    if isinstance(to_kp, str):
        if not to_kp in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(to_kp))
        else:
            to_kp = config.KEYPOINTS.index(to_kp)

    edge_list = config.EDGES + [edge[::-1] for edge in config.EDGES]
    edge_id = edge_list.index((from_kp, to_kp))
    
    if centers is None:
        kp = get_keypoints(heatmaps)
        kp = [k for k in kp if k['id']==from_kp]
        centers = [k['xy'].tolist() for k in kp]

    kp_offsets = offsets[:,:,2*edge_id:2*edge_id+2]
    # dists = np.zeros(offsets.shape[:2]+(len(centers),))
    masks = np.zeros(offsets.shape[:2]+(len(centers),), dtype='bool')
    idx = np.rollaxis(np.indices(offsets.shape[1::-1]), 0, 3).transpose((1,0,2))
    for j, c in enumerate(centers):
        dists = np.sqrt(np.square(idx-c).sum(axis=-1))
        dists_x = np.abs(idx[:,:,0] - c[0])
        dists_y = np.abs(idx[:,:,1] - c[1])
        masks[:,:,j] = (dists<=radius)
        if every > 1:
            d_mask = np.logical_and(np.mod(dists_x.astype('int32'), every)==0, np.mod(dists_y.astype('int32'), every)==0)
            masks[:,:,j] = np.logical_and(masks[:,:,j], d_mask)

    mask = masks.sum(axis=-1) > 0
    # dists = dists.min(axis=-1)
    # mask = dists <= radius
    I, J = np.nonzero(mask)

    if img is not None:
        plt.imshow(img)
        
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 200
    plt.quiver(J, I, kp_offsets[I,J,0], kp_offsets[I,J,1], color='r', angles='xy', scale_units='xy', scale=1)
    plt.savefig(save_path+'middle_offsets.jpg',bbox_inches = 'tight')
    plt.close()

def plot_poses(img, skeletons, save_path=None):

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()
    
    #img = img.astype('uint8')
    canvas = img.copy()

    for i in range(17):
        rgba = np.array(cmap(1 - i/17. - 1./34))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            cv.circle(canvas, tuple(skeletons[j][i, 0:2].astype('int32')), 2, colors[i], thickness=-1)

    to_plot = cv.addWeighted(img, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:,:,[2,1,0]])
    fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    for i in range(config.NUM_EDGES):
        for j in range(len(skeletons)):
            edge = config.EDGES[i]
            if skeletons[j][edge[0],2] == 0 or skeletons[j][edge[1],2] == 0:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            
    plt.imsave(save_path+'pose.jpg',canvas[:,:,:])
    fig = matplotlib.pyplot.gcf()