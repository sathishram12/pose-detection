
## Edited version originally from here:
# https://github.com/iwyoo/tf-bilinear_sampler/blob/master/bilinear_sampler.py

import io
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from post_proc import *
from plot import *

save_path = './demo_result/' 
def _get_grid_array(N, H, W, h, w):
    N_i = tf.range(N)
    H_i = tf.range(h+1, h+H+1)
    W_i = tf.range(w+1, w+W+1)
    n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
    n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
    h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
    w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
    n = tf.cast(n, tf.float32) # [N, H, W, 1]
    h = tf.cast(h, tf.float32) # [N, H, W, 1]
    w = tf.cast(w, tf.float32) # [N, H, W, 1]

    return n, h, w

def bilinear_sampler(x, v):

  shape = tf.shape(x) # TRY : Dynamic shape
  N = shape[0]
  H_ = H = shape[1]
  W_ = W = shape[2]
  h = w = 0

  
  x = tf.pad(x, ((0,0), (1,1), (1,1), (0,0)), mode='CONSTANT')
  vx, vy = tf.split(v, 2, axis=3)
  n, h, w = _get_grid_array(N, H, W, h, w) # [N, H, W, 3]

  vx0 = tf.floor(vx)
  vy0 = tf.floor(vy)
  vx1 = tf.math.ceil(vx)
  vy1 = tf.math.ceil(vy) # [N, H, W, 1]

  iy0 = vy0 + h
  iy1 = vy1 + h
  ix0 = vx0 + w
  ix1 = vx1 + w

  H_f = tf.cast(H_, tf.float32)
  W_f = tf.cast(W_, tf.float32)
  mask = tf.less(ix0, 1)
  mask = tf.logical_or(mask, tf.less(iy0, 1))
  mask = tf.logical_or(mask, tf.greater(ix1, W_f))
  mask = tf.logical_or(mask, tf.greater(iy1, H_f))

  iy0 = tf.where(mask, tf.zeros_like(iy0), iy0)
  iy1 = tf.where(mask, tf.zeros_like(iy1), iy1)
  ix0 = tf.where(mask, tf.zeros_like(ix0), ix0)
  ix1 = tf.where(mask, tf.zeros_like(ix1), ix1)


  i00 = tf.concat([n, iy0, ix0], 3)
  i01 = tf.concat([n, iy1, ix0], 3)
  i10 = tf.concat([n, iy0, ix1], 3)
  i11 = tf.concat([n, iy1, ix1], 3) # [N, H, W, 3]
  i00 = tf.cast(i00, tf.int32)
  i01 = tf.cast(i01, tf.int32)
  i10 = tf.cast(i10, tf.int32)
  i11 = tf.cast(i11, tf.int32)

  x00 = tf.gather_nd(x, i00)
  x01 = tf.gather_nd(x, i01)
  x10 = tf.gather_nd(x, i10)
  x11 = tf.gather_nd(x, i11)

  dx = tf.cast(vx - vx0, tf.float32)
  dy = tf.cast(vy - vy0, tf.float32)
  
  w00 = (1.-dx) * (1.-dy)
  w01 = (1.-dx) * dy
  w10 = dx * (1.-dy)
  w11 = dx * dy
  
  output = tf.add_n([w00*x00, w01*x01, w10*x10, w11*x11])

  return output


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def get_jet_color(v, vmin, vmax):
    c = np.zeros(3)
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin
    if v < (vmin + 0.125 * dv):
        c[0] = 256 * (0.5 + (v * 4))  # B: 0.5 ~ 1
    elif v < (vmin + 0.375 * dv):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4  # G: 0 ~ 1
    elif v < (vmin + 0.625 * dv):
        c[0] = 256 * (-4 * v + 2.5)  # B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375))  # R: 0 ~ 1
    elif v < (vmin + 0.875 * dv):
        c[1] = 256 * (-4 * v + 3.5)  # G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5)  # R: 1 ~ 0.5
    return c


def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x, :] = get_jet_color(gray_img[y, x], 0, 1)
    return out

def dump_output(heatmaps, shortoffset, midoffset ):
    scores        = heatmaps[0].flatten()
    short_offsets = shortoffset[0].flatten()
    mid_offsets   = midoffset[0].flatten()

    with open('posenet_decoder_deep_dive/image_sample_data.h', 'w') as fout:
        print('#include <stdint.h>\n',file=fout)
        print('static const float heatmap[] = {', file=fout)
        scores.tofile(fout, ', ', '%f')
        print('};\n', file=fout)
    
        print('\nstatic const float short_offsets[] = {', file=fout)
        short_offsets.tofile(fout, ', ', '%f')
        print('};\n', file=fout)
    
        print('\nstatic const float mid_offsets[] = {', file=fout)
        mid_offsets.tofile(fout, ', ', '%f')
        print('};\n', file=fout)

def probe_model(model, test_img_path):
    img = cv2.imread(test_img_path)  # B,G,R order
    img = cv2.resize(img, (config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[0])) 
    input_img = img[np.newaxis, :, :, [2, 1, 0]]
    inputs = tf.convert_to_tensor(input_img)
    output_blobs = model.predict(inputs)
    
    kp_maps = output_blobs[0]
    short_offset = output_blobs[1]
    mid_offset = output_blobs[2]
    H = compute_heatmaps(kp_maps[0], short_offset[0])
    for i in range(config.NUM_KP):
        H[:,:,i] = gaussian_filter(H[:,:,i], sigma=2)
    plt.imsave(save_path+'heatmaps.jpg', H[:,:,config.KEYPOINTS.index('Rshoulder')]*10)
    visualize_short_offsets(offsets=short_offset[0], heatmaps=H, keypoint_id='Rshoulder', img=img, every=8,save_path=save_path)
    visualize_mid_offsets(offsets= mid_offset[0], heatmaps=H, from_kp='Rshoulder', to_kp='Rhip', img=img, every=8,save_path=save_path)
    pred_kp = get_keypoints(H)
    pred_skels = group_skeletons(keypoints=pred_kp, mid_offsets=mid_offset[0])
    pred_skels = [skel for skel in pred_skels if (skel[:,2]>0).sum() > 4]
    print ('Number of detected skeletons: {}'.format(len(pred_skels)))
    plot_poses(img, pred_skels,save_path=save_path)
    dump_output(output_blobs[3],  output_blobs[4],  output_blobs[5])
    
    figure = plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1, title='kp maps')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img[:,:,[2,1,0]])
    plt.imshow(kp_maps[0, :, :, 6], alpha = 0.3)

    plt.subplot(2, 2, 2, title='short offset loss')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img[:,:,[2,1,0]])
    plt.imshow(short_offset[0, :, :, 13], alpha = 0.3)

    plt.subplot(2, 2, 3, title='kp maps (No Bilinear)')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(output_blobs[3][0, :, :, 0])

    plt.subplot(2, 2, 4, title='short offset loss (No Bilinear)')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(output_blobs[4][0, :, :, 0])

    return figure
