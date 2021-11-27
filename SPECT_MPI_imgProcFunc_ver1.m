function data = SPECT_MPI_imgProcFunc_ver1(img)
% SPECT_MPI_imgProcFunc_type1
% centroids: calculation is based on red channel
% mask: calculation is based on red-thresholded image
% registration: 3-dimensional, using masked-red thresholded image

addpath('./helperFunctions');
img = cc_img(img); % ccimg size = 712x890x3

%% step 1: calculating the centroid of LVC
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
% masking the image, with a mild red color thresholding
masked_img = img;
masked_img(repmat(masked_img(:,:,1)<=20, [1,1,3])) = 0; % very-mild red color thresholding
masked_img(~repmat(mask,[1,1,3])) = 0;

%% step 2: 3D-registration
% 3D registration perfrom on masked image
tmp = regist_3d(to3d(rgb2gray(masked_img)));
data = cat(4, tmp(:,:,1:40), tmp(:,:,41:80)); % ch. 1: stress, ch. 2: rest

end
