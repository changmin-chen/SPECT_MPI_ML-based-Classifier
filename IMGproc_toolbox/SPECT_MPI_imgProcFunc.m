function data = SPECT_MPI_imgProcFunc(img, version)
% SPECT_MPI_imgProcFunc_ver 0
% remove redundant things on pictures, like white bars and numbers
% this function turn RGB into grayscale, and concatenate the blocks as 3D volume
% ***no further processing steps were performed***
% ------
% SPECT_MPI_imgProcFunc_ver 1
% centroids: calculation is based on red channel
% mask: calculation is based on red-thresholded image
% registration: 3-dimensional
% regist. estimation: masked-red-thresholded image
% regist. application: masked image
% -----
% SPECT_MPI_imgProcFunc_ver 2
% also 3D registration, but dose not perform masking

%% Basic image processing
img = cc_img(img); % ccimg size = 712x890x3

%% Furthur image processing (for ver1 or ver2)
switch version
    case 'ver1'
        fprintf('Perform image processing verion 1:\n')
        img = regist3d_estimate_and_reslice(img);
        img = mask_infwall(img, 'mirror');
    case 'ver2'
        fprintf('Perform image processing verion 2:\n')
        img = regist3d_estimate_and_reslice(img);
    otherwise
         fprintf('Perform image processing verion 0:\n')
end

%% Output
img = to3d(rgb2gray(img));
data = uint8(cat(4, img(:,:,1:40), img(:,:,41:80))); % ch. 1: stress, ch. 2: rest

end
