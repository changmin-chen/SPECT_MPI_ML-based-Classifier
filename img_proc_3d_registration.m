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

%% step 2: calculating the centroid of LVC
% mask evaluation is based on threshold red 165
lvc_threshold = 165;
lvc = img(:,:,1) > lvc_threshold;

% finding the centroid of LVC, evaluation is based on red channel only
wall = to3d(img(:,:,1));
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
% pixel number analysis
% figure, plot(1:40, stress_centroids(:,end)), title('number of pixels for stress')
% figure, plot(1:40, rest_centroids(:,end)), title('number of pixels for rest')

% plot the centroids and the half-circle masks
% show(lvc, ['binarized heart wall, threshold red ', num2str(lvc_threshold)]);
% h1 = gca; hold(h1, 'on')
% count = 1; % stress centroids
% for i = 1:2:7
%     for j = 1:10
%         plot((j-1)*89+stress_centroids(count,2), ((i-1)*89+stress_centroids(count,1)), 'or', 'Parent', h1)
%         count = count+1;
%     end
% end
% count = 1; % rest centroids
% for i = 2:2:8
%     for j = 1:10
%         plot((j-1)*89+rest_centroids(count,2), ((i-1)*89+rest_centroids(count,1)), 'og', 'Parent', h1)
%         count = count+1;
%     end
% end
% hold(h1, 'off')

% get mask and show masking performance
mask = get_mask(lvc, stress_centroids, rest_centroids);
figure, imshow(mask, []), title('masks')
% figure, imshow(lvc, []), title('non-masked LVC')
% figure, imshow(lvc.*mask, []), title('masked LVC')

% show masked image
masked_img = img;
masked_img(repmat(masked_img(:,:,1)<80, [1,1,3])) = 0; % red color thresholding
masked_img(~repmat(mask,[1,1,3])) = 0;
show(masked_img, 'masked, color-thresholded image');

%% step 3: 3D-registration
% masking the LVC wall
masked_wall = to3d(rgb2gray(masked_img));
masked_stress_wall = masked_wall(:,:,1:40);
masked_rest_wall = masked_wall(:,:,41:80);

% estimate registration: stress is fixed, rest is moving
[optimizer,metric] = imregconfig('monomodal');
tform_SA = imregtform(masked_rest_wall(:,:,1:20),masked_stress_wall(:,:,1:20),...
    'affine',optimizer,metric);
tform_HLA = imregtform(pad_wall(masked_rest_wall(:,:,21:30)), pad_wall(masked_stress_wall(:,:,21:30)),...
    'affine',optimizer,metric);
tform_VLA = imregtform(pad_wall(masked_rest_wall(:,:,31:40)), pad_wall(masked_stress_wall(:,:,31:40)),...
    'affine',optimizer,metric);

% perform registration on original image using affine transformations
img3d = to3d(rgb2gray(img)); % registrated, original
% img3d = to3d(rgb2gray(masked_img)); % registrated, masked
stress3d = img3d(:,:,1:40);
rest3d = img3d(:,:,41:80);
rest3d(:,:,1:20) = imwarp(rest3d(:,:,1:20), tform_SA, 'OutputView', imref3d([89, 89, 20]));
rest3d(:,:,21:30) = imwarp(rest3d(:,:,21:30), tform_SA, 'OutputView', imref3d([89, 89, 10]));
rest3d(:,:,31:40) = imwarp(rest3d(:,:,31:40), tform_SA, 'OutputView', imref3d([89, 89, 10]));
data = cat(4, stress3d, rest3d);
niftiwrite(data, 'proc_data.nii');

% show registration performance
show(toccimg(cat(3, stress3d, rest3d)), 'original image, registrated based on masked image')
