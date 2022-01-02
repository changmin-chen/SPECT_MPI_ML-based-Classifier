%% init
clear, clc
close all

src = '../../2038.jpg'; % abnormal, w/ inf. wall excessive signal

img = imread(src); % ccimg size = 712x890x3
show(img, 'original image')

%% crop and concatenated image
img = cc_img(img);
show(img, 'crop and concatenated image')

%% step 1: 3D-registration
img = regist3d_estimate_and_reslice(img);

% show registration performance
show(img, 'registrated image')

%% step 2: calculating the centroid of LVC and mask
% heart wall for display is based on threshold 165 to red channel
lvc_threshold = 165;
lvc = img(:,:,1) > lvc_threshold;

% finding the centroid of LVC, evaluation is based on red channel
[centroids, ~, ~] = get_centroids(img);
stress_centroids = centroids(1:40, :);
rest_centroids = centroids(41:80, :);

% plot the centroids and the half-circle masks
show(lvc, 'binarized heart wall');
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
[~, mask] = mask_infwall(img);
show(mask, 'original mask')
[~, mask_mr] = mask_infwall(img, 'mirror');
show(mask_mr, 'mirrored mask')

% show masked-registered image
img(~repmat(mask_mr,[1,1,3])) = 0;
show(img, 'registrated and masked image');

%% helper functions
function show(img, str)
% display image with title
img = uint8(img);
figure, imagesc(img)
axis off
if size(img,3) == 1
    colormap gray
end
if str
    title(str)
end
end
