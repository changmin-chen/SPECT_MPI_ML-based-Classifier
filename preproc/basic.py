import numpy as np
import cv2

def _image_to3d(image):
    block_size = 90
    vol = np.empty(shape=(block_size,block_size,0), dtype=np.uint8)
    for block_y in range(0, block_size*8, block_size):
        for block_x in range(0, block_size*10, block_size):
            idx_y = np.arange(block_y, block_y + block_size)
            idx_x = np.arange(block_x, block_x + block_size)
            cropped_block = image[idx_y[:,np.newaxis], idx_x, None]
            vol = np.concatenate((vol, cropped_block), axis=2)

    return vol


def _shield_number_labels(vol):
    vol[:16, :18, :] = 0
    vol[:4, :, :] = 0
    vol[:, :4, :] = 0

    return vol


def _stress_rest_divide(vol):
    stress = [vol[:,:,np.concatenate([np.arange(0,10), np.arange(20,30)], axis=0)], # SA view
        vol[:,:,np.arange(40,50)], # HLA view
        vol[:,:,np.arange(60,70)]] # VLA view

    rest = [vol[:,:,np.concatenate([np.arange(10,20), np.arange(30,40)], axis=0)],
        vol[:,:,np.arange(50,60)],
        vol[:,:,np.arange(70,80)]]

    return stress, rest


def image_preproc_basic(image):
    # cut-out desired part of the image
    rows_SA = np.arange(50,411)
    rows_HLA = np.arange(483,664)
    rows_VLA = np.arange(707,888)
    rows = np.concatenate([rows_SA, rows_HLA, rows_VLA], axis=0)
    cols = np.arange(69,970)
    image_cleaned = image[rows[:,np.newaxis], cols]

    # get the red-channel and grayscale-form of the image
    image_red = image_cleaned[:,:,-1] # red channel is the last channel in bgr
    image_gray = cv2.cvtColor(image_cleaned, cv2.COLOR_BGR2GRAY)
    
    # concatenate blocks into 3D volume.
    vol_red = _image_to3d(image_red)
    vol_gray = _image_to3d(image_gray)

    # shield the number labels at the upper-left corner
    vol_red = _shield_number_labels(vol_red)
    vol_gray = _shield_number_labels(vol_gray)

    # divide into two 3D volume according to status: Stress or Rest.
    stress_red, rest_red = _stress_rest_divide(vol_red)
    stress_gray, rest_gray = _stress_rest_divide(vol_gray)

    return stress_red, rest_red, stress_gray, rest_gray