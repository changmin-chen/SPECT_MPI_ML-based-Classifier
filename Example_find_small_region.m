%% init
clear, clc
close all
addpath('./helperFunctions');

% src = '../2038.jpg'; % abnormal, w/ inf. wall excessive signal
src = '../2048.jpg'; % abnormal, w/o ...
% src = '../1002.jpg'; % normal, w/ mild excessive
% src = '../2120.jpg';
% src = '../4008.jpg';

img = cc_img(imread(src)); % ccimg size = 712x890x3

%% original image
% show(img, 'original image')

%% red channel threshod
idx = img(:,:,1) <= 100;
img(repmat(idx,[1,1,3])) = 0;
show(img, 'original image, red threshod 100')

%% Laplacian
kernal = [-1 -1 -1; -1 8 -1; -1 -1 -1];
L = double(rgb2gray(img));
L = conv2(L, kernal, 'same');
img_L = L+double(rgb2gray(img));
Le = L>70;
% show(L, 'Laplacian')
% show(Le, 'edge') % edge

%% region growing
I = ~to3d(Le);
area_threshold = 150;
% show(toccimg(I), '')
for p = 1:80
    Ip = I(:,:,p);
    J_list = [];
    if sum(sum(Ip)) < 89*89
        while sum(sum(Ip))> 0
            [x, y] = find(Ip, 1, 'first');
            J = regiongrowing(Ip, x, y, 0.2);
            Ip(J) = false;
            J_list = cat(3, J_list, J);
        end
        J_area = sum(J_list, 1:2);
        close_idx = J_area < area_threshold;
        J_close = sum(J_list(:,:, close_idx), 3);
        I(:,:,p) = J_close;
    else
        I(:,:,p) = false(size(I, 1:2));
    end
    
end


%% processing the close region
I_close = logical(toccimg(I));
show(I_close, 'to be close, original')

% step 1: dilation
se = strel('diamond', 1); 
I_close = imdilate(I_close, se);

% step 2: neightbor mix
mixed_map = false(size(I_close));
for r = 1:8
    for c = 1:10
        valid = c-1>=1 && c+1<=10;
        if valid
            center =  I_close((r-1)*89+1:r*89, (c-1)*89+1:c*89);
            lt_nei = I_close((r-1)*89+1:r*89, (c-2)*89+1:(c-1)*89);
            rt_nei = I_close((r-1)*89+1:r*89, c*89+1:(c+1)*89);
            mixed = (lt_nei | center) | rt_nei;
            mixed_map((r-1)*89+1:r*89, (c-1)*89+1:c*89) = mixed;
        end
    end
end
I_close = mixed_map;

% step 3: stress-rest mix
for r = 1: 2: 8
    st = I_close((r-1)*89+1: r*89, :);
    rt = I_close(r*89+1: (r+1)*89, :);
    mixed = st | rt;
    I_close((r-1)*89+1: (r+1)*89, :) = repmat(mixed, [2,1]);
end


show(I_close, 'to be close, processed')
img(repmat(I_close, [1,1,3])) = 0;
show(img, 'closed img')
