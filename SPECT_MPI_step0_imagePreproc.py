import numpy as np
import cv2
import nibabel as nib
import glob2 as gb
from os.path import join, splitext, split
from pathlib import Path
from preproc.basic import image_preproc_basic
from preproc.register import registration_estimate, imregister
from preproc.masking import compute_cetroids, get_vol_mask


class SPECT_MPI_image():
    def __init__(self, directory, image_proc_version=0):
        self.arg = {'wall_threshold_initial': 165,
                'wall_lw_limit': 500,
                'wall_up_limit': 1800,
                'wall_threshold_lower': 20,
                'wall_threshold_higher': 200,
                'mask_overlap_threshold': 0.75/2,
                'mask_y_threshold': 80}
        data = cv2.imread(directory)
        self.stress_red, self.rest_red, self.stress_gray, self.rest_gray = image_preproc_basic(data)
        self.image_proc_version = image_proc_version
        self.output = None

    def register(self):
        for i in range(3):
            _, final_transform, _ = registration_estimate(self.stress_gray[i], self.rest_gray[i])
            self.rest_gray[i] = imregister(self.stress_gray[i], self.rest_gray[i], final_transform)
            self.rest_red[i] = imregister(self.stress_red[i], self.rest_red[i], final_transform)

    def masking(self):
        for i, (vol_stress,vol_rest) in enumerate(zip(self.stress_red, self.rest_red)):
            centroid_stress = compute_cetroids(vol_stress, self.arg)
            centroid_rest = compute_cetroids(vol_rest, self.arg)
            mask_stress = get_vol_mask(vol_stress, centroid_stress, self.arg)
            mask_rest = get_vol_mask(vol_rest, centroid_rest, self.arg)
            joined_mask = np.logical_or(mask_stress, mask_rest)
            self.stress_gray[i] = self.stress_gray[i]*joined_mask
            self.rest_gray[i] = self.rest_gray[i]*joined_mask

    def acquire_final_output(self):
        stress_final = np.concatenate(self.stress_gray, axis=2)
        rest_final = np.concatenate(self.rest_gray, axis=2)
        output = np.concatenate((stress_final[:,:,:,None], rest_final[:,:,:,None]), axis=3)
        self.output = nib.Nifti1Image(output, np.eye(4))

    def execute(self):
        if self.image_proc_version==1:
            self.register()
            self.masking()
        elif self.image_proc_version==2:
            self.register()

        self.acquire_final_output()
            

def main(image_proc_version):
    # directories to read the images
    train_paths = gb.glob(join('..', 'data', 'TrainSet', '*.jpg'))
    test_paths = gb.glob(join('..', 'data', 'TestSet', '*.jpg'))

    # directories to save the processed images
    save_train = join('..', 'proc_data', 'TrainSet')
    save_test = join('..', 'proc_data', 'TestSet')
    Path(save_train).mkdir(parents=True, exist_ok=True)
    Path(save_test).mkdir(parents=True, exist_ok=True)

    # processing
    for i in range(len(train_paths)):
        data = SPECT_MPI_image(train_paths[i], image_proc_version)
        data.execute()
        sbjID = splitext(split(train_paths[i])[1])[0]
        nib.save(data.output, join(save_train, sbjID))

    for i in range(len(test_paths)):
        data = SPECT_MPI_image(test_paths[i], image_proc_version)
        data.execute()
        sbjID = splitext(split(train_paths[i])[1])[0]
        nib.save(data.output, join(save_test, sbjID))


if __name__ == '__main__':
    image_proc_version = 0
    main(image_proc_version)