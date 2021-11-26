%% init
clear, clc
close all

% src = '../2038.jpg'; % abnormal, with inf. wall excessive signal
src = '../2048.jpg'; % abnormal, w/o ...
img = cc_img(imread(src)); % ccimg size = 712x890x3

%% original image
show(img, 'original image')

%% step 1: color thresholding
color_thres_img1 = img;
threshold_stage1 = 110;
idx_stage1 = img(:,:,1) <= threshold_stage1;
color_thres_img1(repmat(idx_stage1,[1,1,3])) = 0;
show(color_thres_img1, 'threshold red 110')

color_thres_img2 = img;
threshold_stage2 = 185;
idx_stage2 = img(:,:,1) <= threshold_stage2;
color_thres_img2(repmat(idx_stage2,[1,1,3])) = 0;
show(color_thres_img2, 'threshold red 185')

%% step 2: calculating the centroid of LVC
% finding the centroid of LVC, thresholding is operated on red channel only
wall = to3d(img(:,:,1));
stress_wall = wall(:,:,1:40); % stress
stress_centroids = round(get_centroids(stress_wall));
rest_wall = wall(:,:,41:80); % rest
rest_centroids = round(get_centroids(rest_wall));

% plot the centroids
lvc_threshold = 165; 
idx_lvc = img(:,:,1) <= lvc_threshold;
show(~idx_lvc, ['binarized heart wall, threshold red ', num2str(lvc_threshold)]);
h = gca;
hold(h, 'on')

count = 1;
for i = 1:2:7 % stress
    for j = 1:10
        plot((j-1)*89+stress_centroids(count,2), ((i-1)*89+stress_centroids(count,1)), 'or', 'Parent', h)
        count = count+1;
    end
end
count = 1;
for i = 2:2:8 % rest
    for j = 1:10
        plot((j-1)*89+rest_centroids(count,2), ((i-1)*89+rest_centroids(count,1)), 'og', 'Parent', h)
        count = count+1;     
    end
end

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

%% helper functions

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

% helper func2: crop and concatenate image
function C = cc_img(img)
% crop and concatenate binary image
% output 2d cropped and concatenated image
% original block size = 90x90
% output block size = 89x89

% SA view
SA_lu = [51, 70];
SA_rd = [411, 970];
SAr = setxor(SA_lu(1):SA_rd(1), SA_lu(1):90:SA_rd(1));
SAc = setxor(SA_lu(2):SA_rd(2), SA_lu(2):90:SA_rd(2));
% HLA view
HLA_lu = [484, 70];
HLA_rd = [664, 970];
HLAr = setxor(HLA_lu(1):HLA_rd(1), HLA_lu(1):90:HLA_rd(1));
HLAc = setxor(HLA_lu(2):HLA_rd(2), HLA_lu(2):90:HLA_rd(2));
% VLA view
VLA_lu = [708, 70];
VLA_rd = [888, 970];
VLAr = setxor(VLA_lu(1):VLA_rd(1), VLA_lu(1):90:VLA_rd(1));
VLAc = setxor(VLA_lu(2):VLA_rd(2), VLA_lu(2):90:VLA_rd(2));
% concatenate
SA = img(SAr,SAc,:);
HLA = img(HLAr,HLAc,:);
VLA = img(VLAr,VLAc,:);
C = cat(1, SA, HLA, VLA);

% remove number
proc_area = [14, 17];
for j = 1:10
    for i = 1:8
        C((i-1)*89+1: (i-1)*89+proc_area(1), (j-1)*89+1: (j-1)*89+proc_area(2), :) = 0;
    end
end

end

% helper func3: ccimg to 3D
function D = to3d(ccimg)
% cc_img block size = 89x89, total size = 712x890
if any(size(ccimg,1:3)~=[712,890,1])
    error('Image should be binary, grayscale or single-channel with block size 89x89.')
end
D = zeros(89, 89, 80);
count = 1;
for i = 1:8
    for j = 1:10
        D(:,:,count) = ccimg((i-1)*89+1: i*89, (j-1)*89+1: j*89);
        count = count+1;
    end
end
% ordering, D(:,:,1:40) = stress and D(:,:,41:80) = rest
stress_idx = [1:10, 21:30, 41:50, 61:70];
rest_idx = stress_idx +10;
D(:,:,1:80) = D(:,:,[stress_idx, rest_idx]);

end

% helper func 4: 3D to ccimg
function ccimg = toccimg(D)
% 3D volume  size = 89x89x80, (rest:1:40, stress:41:80)
if any(size(D,1:3)~=[89,89,80])
    error('Image should be 3D with size 89x89x80.')
end
% ordering
D(:,:,1:80) = D(:,:,[1:10, 41:50, 11:20, 51:60, 21:30, 61:70, 31:40, 71:80]);
ccimg = zeros(712, 890);
count = 1;
for i = 1:8
    for j = 1:10
        ccimg((i-1)*89+1: i*89, (j-1)*89+1: j*89) = D(:,:, count);
        count = count+1;
    end
end
end

% helper func 5: get centroids of each blocks
function centroids = get_centroids(wall)
% threshold setting: 20 if too small, 165 is gold standard, 200 if excessive
% input:
% wall: red channel
if any(size(wall)~=[89, 89, 40])
    error('input matrix size should be 89x89x40. Please assign the rest and stress volumes separately')
elseif isa(wall, 'logical')
    error('input should be Red-channel, not binary mask.')
end

tmp = []; % temporarily save the centroids of part of the blocks
[x, y] = meshgrid(1:89, 1:89); % coordinates for centroid calculation
threshold = 165; % the default threshold (may be adjusted)
lower_limit = 500; % lower-limit wall pixel number threshold
upper_limit = 2300; % upper-limit wall pixel number threshold

for p = 1: 40
    pwall = wall(:,:,p);
    
    % testing the better threshold for the block element
    if sum(sum(logical(pwall>threshold))) < lower_limit  
        threshold = 20;  % adjusted threshold
    elseif sum(sum(logical(pwall>threshold))) > upper_limit 
        threshold = 200; % adjusted threshold
    end
    
    % wall size after threshold adjustion
    if sum(sum(logical(pwall>threshold))) < lower_limit || sum(sum(logical(pwall>threshold))) > upper_limit
        continue % move to the next plane if the wall size still can't meet the requirement
    else
        pwall = pwall>threshold;
        center_row = mean(y(pwall));
        center_col = mean(x(pwall));
        tmp = [tmp;...
            center_row, center_col, p];
    end
end
SA_centroid = mean(tmp(tmp(:,3)<=20, 1:2), 1); % whole-world SA centroid
HLA_centroid = mean(tmp((tmp(:,3)>=21) & (tmp(:,3)<=30), 1:2), 1); % whole-world HLA centroid
VLA_centroid = mean(tmp(tmp(:,3)>=31, 1:2), 1);  % whole-world VLA centroid
centroids = zeros(40, 3); % save the final decided centroids of all blocks
centroids(tmp(:,3), :) = tmp; % assign the centroid for each block element if available

% assign the missed-block centroids as the whole-world centroid
miss_blocks = setxor(1:40, tmp(:,3))';
for p = miss_blocks
    if p<=20
        centroids(p, :) = [SA_centroid, p];
    elseif (p>=21) && (p<=30)
        centroids(p, :) = [HLA_centroid, p];
    else
        centroids(p, :) = [VLA_centroid, p];
    end
end

end
