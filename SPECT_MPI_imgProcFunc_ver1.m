function data = SPECT_MPI_imgProcFunc_ver1(img)
% SPECT_MPI_imgProcFunc_ver 1
% centroids: calculation is based on red channel
% mask: calculation is based on red-thresholded image
% registration: 3-dimensional
% regist. estimation: masked-red-thresholded image
% regist. application: masked image

addpath('./helperFunctions');
img = cc_img(img); % ccimg size = 712x890x3

%% step 1: 3D-registration
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

%% step 2: calculating the centroid of LVC and the mask of the heart wall
% finding the centroid of LVC, evaluation is based on red channel
wall = to3d(img(:,:,1)); % red channel
stress_wall = wall(:,:,1:40); % get stress centroids
stress_centroids = [...
    round(get_centroids(stress_wall(:,:,1:20), 500, 1800));...
    round(get_centroids(stress_wall(:,:,21:30), 500, 1800));...
    round(get_centroids(stress_wall(:,:,31:40), 500, 1800));
    ];
rest_wall = wall(:,:,41:80); % get rest centroids
rest_centroids = [...
    round(get_centroids(rest_wall(:,:,1:20), 500, 1800));...
    round(get_centroids(rest_wall(:,:,21:30), 500, 1800));...
    round(get_centroids(rest_wall(:,:,31:40), 500, 1800));
    ];

% get mask and show masking performance, mask evaluation is based on threshold red 165
lvc_threshold = 165;
lvc = img(:,:,1) > lvc_threshold;
mask = get_mask(lvc, stress_centroids, rest_centroids);
mask = mask_mirroring(mask);

% masking the image
img(~repmat(mask,[1,1,3])) = 0;

%% output
img = to3d(rgb2gray(img));
data = uint8(cat(4, img(:,:,1:40), img(:,:,41:80))); % ch. 1: stress, ch. 2: rest

end
