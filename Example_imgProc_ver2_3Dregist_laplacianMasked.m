%% init
clear, clc
close all
addpath('./helperFunctions');

% src = '../2038.jpg'; % abnormal, w/ inf. wall excessive signal
% src = '../2048.jpg'; % abnormal, w/o ...
% src = '../1002.jpg'; % normal, w/o
% src = '../1003.jpg'; % normal, w/o
% src = '../1004.jpg'; % normal, w/o
src = '../2120.jpg'; % abnormal, w/

img = cc_img(imread(src)); % ccimg size = 712x890x3

%% original image
show(img, 'original image')

%% step 2: 3D-registration
% estimation
[~, tforms] = regist_3d(to3d(rgb2gray(img)));

% application (channel by channel)
for ch = 1:3 % R, G and B
    
    % extract image channel by channel
    img_ch = to3d(img(:,:,ch));   
    stress = img_ch(:,:,1:40);
    rest = img_ch(:,:,41:80);
    
    % rest volume is registered to stress volume
    rest(:,:,1:20) = imwarp(rest(:,:,1:20), tforms{1}, 'OutputView', imref3d([89, 89, 20]));
    rest(:,:,21:30) = imwarp(rest(:,:,21:30), tforms{2}, 'OutputView', imref3d([89, 89, 10]));
    rest(:,:,31:40) = imwarp(rest(:,:,31:40), tforms{3}, 'OutputView', imref3d([89, 89, 10]));
    
    % write the registered image to that channel
    img_ch_registed = cat(3, stress, rest);
    img(:,:, ch) = toccimg(img_ch_registed);
end

% show registration performance
show(img, 'registrated image')

%% step 2: Get small region mask
mask = get_La_mask(img);
show(mask, 'mask, original')

% mask processing
mask_proc = mask;

% close
se = strel('diamond', 4); 
mask_proc = imdilate(~mask_proc, se);
mask_proc = ~mask_proc;
show(mask_proc, 'mask closed')

% mirror
mask_proc = mask_mirroring(mask_proc);
show(mask_proc, 'mask closed and mirrored')

img(~repmat(mask_proc, [1,1,3])) = 0;
show(img, 'masked image')

