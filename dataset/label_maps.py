import numpy as np
import cv2

from config import config

class PredictionData:
    
    def __init__(self, keypoints):
        self.map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
        self.idx       = np.rollaxis(np.indices(self.map_shape[::-1]), 0, 3).transpose((1,0,2))
        self.keypoints = keypoints
        self.r         = config.KP_RADIUS
    

    def __get_keypoint_discs(self):
        discs = [[] for _ in range(len(self.keypoints))]
        for i in range(config.NUM_KP):
            centers = [kp[i] for kp in self.keypoints if kp[i][0] != 0  and kp[i][1] != 0]
            dists = np.zeros(self.map_shape+(len(centers),))
            for k, center in enumerate(centers):
                dists[:,:,k] = np.sqrt(np.square(center-self.idx).sum(axis=-1))
            if len(centers) > 0:
                inst_id = dists.argmin(axis=-1)
            count = 0
            for j in range(len(self.keypoints)):
                if self.keypoints[j][i][0] != 0 and self.keypoints[j][i][1] != 0:
                    discs[j].append(np.logical_and(inst_id==count, dists[:,:,count]<=config.KP_RADIUS))
                    count +=1
                else:
                    discs[j].append(np.array([]))
        return discs

    def kp_heatmaps(self):
        self.discs  = self.__get_keypoint_discs()
        kp_maps = np.zeros(self.map_shape+(config.NUM_KP,))
        for i in range(config.NUM_KP):
            for j in range(len(self.discs)):
                if self.keypoints[j][i][0] != 0 and self.keypoints[j][i][1] != 0:
                    kp_maps[self.discs[j][i], i] = 1.
        return kp_maps


    def compute_short_offsets(self):
        x = np.tile(np.arange(self.r, -self.r-1, -1), [2 * self.r +1, 1])
        y = x.transpose()
        m = np.sqrt(x*x + y*y) <= self.r
        kp_circle = np.stack([x, y], axis=-1) * np.expand_dims(m, axis=-1)
    
        offsets = np.zeros(self.map_shape+(2*config.NUM_KP,))
        for i in range(config.NUM_KP):
            for j in range(len(self.keypoints)):
                if self.keypoints[j][i][0] != 0 and self.keypoints[j][i][1] != 0:
                    offsets[:, :,[i, config.NUM_KP+i]]  = self.__copy_with_border_check(
                                                          offsets[:, :,[i, config.NUM_KP+i]], 
                                                          (int(self.keypoints[j][i][0]), 
                                                           int(self.keypoints[j][i][1])), 
                                                            self.discs[j][i], kp_circle)
        
        return offsets
    
    def __copy_with_border_check(self, map, center, disc, kp_circle):
        from_top = max(self.r - center[1], 0)
        from_left = max(self.r - center[0], 0)
        from_bottom = max(self.r -(self.map_shape[0] - center[1]) + 1, 0)
        from_right =  max(self.r - (self.map_shape[1] - center[0]) + 1, 0)
        
        cropped_disc = disc[center[1] - self.r + from_top :center[1] + self.r + 1 - from_bottom,
                            center[0] - self.r + from_left:center[0] + self.r + 1 - from_right]
        
        map[center[1] - self.r + from_top :center[1] + self.r + 1 - from_bottom, 
            center[0] - self.r + from_left:center[0] + self.r + 1 - from_right, :][cropped_disc,:] = \
            kp_circle[from_top:2 * self.r + 1 - from_bottom, from_left:2 * self.r + 1 - from_right, :][cropped_disc,:]
        return map



    def compute_mid_offsets(self):
        offsets = np.zeros(self.map_shape + (4 * config.NUM_EDGES,))
        for i, edge in enumerate((config.EDGES + [edge[::-1] for edge in config.EDGES])):
            for j in range(len(self.keypoints)):
                if self.keypoints[j][edge[0]][0] != 0 and \
                   self.keypoints[j][edge[0]][1] != 0 and \
                   self.keypoints[j][edge[1]][0] != 0 and \
                   self.keypoints[j][edge[1]][1] != 0:
                    
                    m = self.discs[j][edge[0]]
                    dists = [[ self.keypoints[j][edge[1]][0], 
                               self.keypoints[j][edge[1]][1] ]] - self.idx[m,:]
                    offsets[m, 2 * i : 2 * i + 2] = dists
                    
        return offsets
