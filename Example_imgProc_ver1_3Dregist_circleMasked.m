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

% show registration performance
show(img, 'registrated image')

%% step 2: calculating the centroid of LVC and mask
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
show(lvc, ['binarized heart wall, threshold red ', num2str(lvc_threshold)]);
h1 = gca; hold(h1, 'on')
count = 1; % stress centroids
for i = 1:2:7
    for j = 1:10
        plot((j-1)*89+stress_centroids(count,2), ((i-1)*89+stress_centroids(count,1)), 'or', 'Parent', h1)
        count = count+1;
    end
end
count = 1; % rest centroids
for i = 2:2:8
    for j = 1:10
        plot((j-1)*89+rest_centroids(count,2), ((i-1)*89+rest_centroids(count,1)), 'og', 'Parent', h1)
        count = count+1;
    end
end
hold(h1, 'off')

% get mask and show masking performance
mask = get_mask(lvc, stress_centroids, rest_centroids);
show(mask, 'masks, original')
mask = mask_mirroring(mask);
show(mask, 'masks, mirrored')

% show masked-registered image
img(~repmat(mask,[1,1,3])) = 0;
show(img, 'masked image');
