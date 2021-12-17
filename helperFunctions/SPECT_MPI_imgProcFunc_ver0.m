function data = SPECT_MPI_imgProcFunc_ver0(img)
% SPECT_MPI_imgProcFunc_ver 0
% remove redundant things on pictures, like white bars and numbers
% this function turn RGB into grayscale, and concatenate the blocks as 3D volume
% no other processing steps were performed

addpath('./helperFunctions');

tmp = to3d(cc_img(rgb2gray(img)));
data = uint8(cat(4, tmp(:,:,1:40), tmp(:,:,41:80))); % ch. 1: stress, ch. 2: rest

end