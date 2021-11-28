%% init
clear, clc
close all
addpath('./helperFunctions');

src = '../2038.jpg'; % abnormal, w/ inf. wall excessive signal
% src = '../2048.jpg'; % abnormal, w/o ...
% src = '../1002.jpg'; % normal, w/o
% src = '../2120.jpg';

img = cc_img(imread(src)); % ccimg size = 712x890x3

%% original image
show(img, 'original image')

%% step 1: Get small region mask
I_close = get_small_region(img);
img_closed = img;
img_closed(repmat(I_close, [1,1,3])) = 0;
show(img_closed, 'image closed')

%% step 2: 3D-registration
% registration is estimated on closed images, however, registration is perfromed on original image
perf_img = to3d(rgb2gray(img_closed));
[~, tforms] = regist_3d(perf_img);

tmp = to3d(rgb2gray(img));
stress = tmp(:,:,1:40);
rest = tmp(:,:,41:80);
rest(:,:,1:20) = imwarp(rest(:,:,1:20), tforms{1}, 'OutputView', imref3d([89, 89, 20]));
rest(:,:,21:30) = imwarp(rest(:,:,21:30), tforms{2}, 'OutputView', imref3d([89, 89, 10]));
rest(:,:,31:40) = imwarp(rest(:,:,31:40), tforms{3}, 'OutputView', imref3d([89, 89, 10]));

% show registration performance
show(toccimg(cat(3, stress, rest)), 'registrated image')

%% output
stress = uint8(stress);
rest = uint8(rest);
I_close = uint8(to3d(I_close));
data = cat(4, stress, rest, I_close(:,:,1:40)); % because I_close for stress and rest are same, we only save one.
% niftiwrite(data, 'proc_data')

