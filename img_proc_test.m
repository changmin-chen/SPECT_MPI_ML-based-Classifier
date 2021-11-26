%% init
clear, clc
close all

src = '../2048.jpg'; % abnormal
img = imread(src);

%% original image
show(img, 'original image')

%% step 1: color thresholding
color_thres_img1 = img;
threshold_stage1 = 110;
idx_stage1 = img(:,:,1) <= threshold_stage1;
color_thres_img1(repmat(idx_stage1,[1,1,3])) = 0;
show(color_thres_img1, 'color threshold, red 110')

color_thres_img2 = img;
threshold_stage2 = 185;
idx_stage2 = img(:,:,1) <= threshold_stage2;
color_thres_img2(repmat(idx_stage2,[1,1,3])) = 0;
show(color_thres_img2, 'color threshold, red 185')

%% step 2: calculating the centroid of LVC
lvc_threshold = 165;
lvc_img = img;
idx_lvc = img(:,:,1) <= lvc_threshold;
lvc_img(repmat(idx_lvc,[1,1,3])) = 0;
% show(lvc_img, 'LVC removed, demo only')

% finding centroid
hwall = uint8(~idx_lvc);
show(cc_bimg(hwall), 'heart wall');
[shwall, rhwall] = to3d(hwall);

%% step 3: 3D-registration
% s = 1:20;
s = 21:30;
d = 10;
tmp = rhwall(:,:,s);
if size(tmp, 3) < 16
    fixed = zeros(89,89,16);
    fixed(:,:,4:13) = tmp;
else
    fixed = tmp;
end
tmp = shwall(:,:,s);
if size(tmp, 3) < 16
    moving = zeros(89,89,16);
    moving(:,:,4:13) = tmp;
else
    moving = tmp;
end

figure, 
imshowpair(moving(:,:,d), fixed(:,:,d))
title('Before registration')

[optimizer,metric] = imregconfig('monomodal');
% movingRegisteredDefault = imregister(moving,fixed,'affine',optimizer,metric);
figure,
imshowpair(movingRegisteredDefault(:,:,d),fixed(:,:,d))
title('A: Default Registration')
tform = imregtform(moving,fixed,'rigid',optimizer,metric);
tform.T

%% helper func

% helper func1: display image
function show(img, str)
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

% helper func2: crop and concatenate
function C = cc_bimg(img)
% crop and concatenate binary image
% output 2d cropped and concatenated image
% original block size = 90x90
% output block size = 89x89

% SA view
SA_lu = [51, 70];
SA_rd = [411, 970];
rdrop = SA_lu(1):90:SA_rd(1); % drop the white bar
cdrop = SA_lu(2):90:SA_rd(2); % drop the white bar
r = setxor(SA_lu(1):SA_rd(1), rdrop);
c = setxor(SA_lu(2):SA_rd(2), cdrop);
SA = img(r,c);

% HLA view
HLA_lu = [484, 70];
HLA_rd = [664, 970];
rdrop = HLA_lu(1):90:HLA_rd(1); % drop the white bar
cdrop = HLA_lu(2):90:HLA_rd(2); % drop the white bar
r = setxor(HLA_lu(1):HLA_rd(1), rdrop);
c = setxor(HLA_lu(2):HLA_rd(2), cdrop);
HLA = img(r,c);

% VLA view
VLA_lu = [708, 70];
VLA_rd = [888, 970];
rdrop = VLA_lu(1):90:VLA_rd(1); % drop the white bar
cdrop = VLA_lu(2):90:VLA_rd(2); % drop the white bar
r = setxor(VLA_lu(1):VLA_rd(1), rdrop);
c = setxor(VLA_lu(2):VLA_rd(2), cdrop);
VLA = img(r,c);

% concatenate
C = cat(1, SA, HLA, VLA);

% remove number
narea = [14, 17];
for j = 1:10
    for i = 1:8
        C((i-1)*89+1: (i-1)*89+narea(1), (j-1)*89+1: (j-1)*89+narea(2)) = 0;
    end
end

end

% helper func3: img to 3D
function [stress, rest] = to3d(img)
% SA prior slices: 5 to 12
% HLA, VLA prior slices: 3 to 8
% bimg block size = 89x89
bimg = cc_bimg(img);
V = zeros(89, 89, 80);
count = 1;
for i = 1:8
    for j = 1:10
        V(:,:,count) = bimg((i-1)*89+1: i*89, (j-1)*89+1: j*89);
        count = count+1;
    end
end
stress = V(:,:, [1:10, 21:30, 41:50, 61:70]);
rest = V(:,:, [11:20, 31:40, 51:60, 71:80]);

end
