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
masked_img(repmat(masked_img(:,:,1)<20, [1,1,3])) = 0; % red color thresholding
masked_img(~repmat(mask,[1,1,3])) = 0;
show(masked_img, 'masked, color-thresholded image');

%% step 3: 3D-registration
% 3D registration perfrom on masked image
perf_img = to3d(rgb2gray(masked_img));
data = regist_3d(perf_img);

% show registration performance
show(toccimg(data), 'registrated masked-thresholded image')
