import numpy as np


def _count_wall_size(block, threshold):
    block_binarized = block > threshold
    block_binarized = block_binarized.astype(np.uint8)
    num_pixel = np.sum(block_binarized, axis=(0,1))

    return num_pixel


def _get_heart_wall(block, arg):
    threshold = arg['wall_threshold_initial']
    # tuning the threshold value
    initial_num_pixel = _count_wall_size(block, threshold)
    if initial_num_pixel < arg['wall_lw_limit']:
        threshold = arg['wall_threshold_lower']
    elif initial_num_pixel > arg['wall_up_limit']:
        threshold = arg['wall_threshold_higher']

    new_num_pixel = _count_wall_size(block, threshold)
    if new_num_pixel<arg['wall_lw_limit'] or new_num_pixel>arg['wall_up_limit']:
        wall = None
    else:
        wall = block > threshold
    
    return wall


def compute_cetroids(vol, arg):
    centroids = []
    # get (x,y) coordinate of centroid for each block
    for block_idx in range(vol.shape[-1]):
        wall = _get_heart_wall(vol[:,:,block_idx], arg)

        if wall is None:
            centroids.append([np.nan, np.nan])
        else:
            x = np.arange(0, wall.shape[0])
            y = np.arange(0, wall.shape[1])
            xv, yv = np.meshgrid(x, y)
            xc = np.mean(xv[wall], axis=0)
            yc = np.mean(yv[wall], axis=0)
            centroids.append(np.asarray([xc, yc]))

    # compute global centroid based on valid centroid coordinates
    centroids_nda = np.asarray(centroids)
    centroids_valid = centroids_nda[~np.isnan(np.sum(centroids_nda, axis=1)), :]
    global_centroid = np.mean(centroids_valid, axis=0)
    for i, centroid in enumerate(centroids):
        if any(np.isnan(centroid)):
            centroids[i] = global_centroid

    return centroids


def _draw_circle_mask(block_binarized, centroid):
    mask = np.zeros(shape=block_binarized.shape, dtype=bool)
    y = np.arange(0, block_binarized.shape[0])
    x = np.arange(0, block_binarized.shape[1])
    xv, yv = np.meshgrid(x, y)
    xc, yc = centroid[1], centroid[0]

    wall_area = np.sum(block_binarized, axis=(0,1)) 
    radius_square = wall_area/np.pi
    circle_grid = (xv-xc)**2+(yv-yc)**2<=radius_square
    mask[circle_grid] = True
    
    return mask


def get_vol_mask(vol, centroids, arg):
    vol_mask = np.ones(shape=vol.shape, dtype=bool)
    vol_binarized = (vol>arg['wall_threshold_initial']).astype(np.bool)
    for i in range(vol_binarized.shape[-1]):
        block_binarized = vol_binarized[:,:,i]
        centroid = centroids[i]
        circle_mask = _draw_circle_mask(block_binarized, centroid)

        # checking whether apply the mask
        wall_area = np.sum(block_binarized)
        if wall_area > 0:
            overlap_ratio = np.sum(np.logical_and(circle_mask,block_binarized)) / wall_area
            lowest_y = np.sum(block_binarized,axis=1).nonzero()[-1][-1]

            if (overlap_ratio>arg['mask_overlap_threshold']) and (centroid[0]>block_binarized.shape[0]/2):
                circle_mask[:np.round(centroid[0]).astype(np.int_), :] = True # only apply half-circle
                vol_mask[:,:,i] = circle_mask
            elif (overlap_ratio<=arg['mask_overlap_threshold']) and lowest_y>=arg['mask_y_threshold']:
                bar_mask = np.ones(shape=block_binarized.shape, dtype=bool)
                bar_mask[-15:,:] = False
                vol_mask[:,:,i] = bar_mask

    return vol_mask
