clear, clc
close all

src = 'D:\SPECT_MPI\2002.jpg'; % abnormal
img_size = [897, 976];
block_sz = [90, 90];

img = imread(src);
img = rgb2gray(img);

figure,
image(img)

%% Processing
% SA view
SA_lu = [51, 70];
SA_rd = [411, 970] - block_sz;

SA = zeros(block_sz(1), block_sz(2), 40);
count = 1;
for i = SA_lu(1): block_sz(1): SA_rd(1)
    for  j = SA_lu(2): block_sz(2): SA_rd(2)
        SA(:,:,count) = img(i: i+block_sz(1)-1, j: j+block_sz(2)-1);
        count = count +1;
    end
end

SA_stress = SA(:,:,[1:10, 21:30]);
SA_rest = SA(:,:,[11:20, 31:40]);


% HLA view
HLA_lu = [484, 70];
HLA_rd = [664, 970] - block_sz;

HLA = zeros(block_sz(1), block_sz(2), 20);
count = 1;
for i = HLA_lu(1): block_sz(1): HLA_rd(1)
    for  j = HLA_lu(2): block_sz(2): HLA_rd(2)
        HLA(:,:,count) = img(i: i+block_sz(1)-1, j: j+block_sz(2)-1);
        count = count +1;
    end
end

HLA_stress = HLA(:,:,1:10);
HLA_rest = HLA(:,:,11:20);

% VLA view
VLA_lu = [708, 70];
VLA_rd = [888, 970] - block_sz;

VLA = zeros(block_sz(1), block_sz(2), 20);
count = 1;
for i = VLA_lu(1): block_sz(1): VLA_rd(1)
    for  j = VLA_lu(2): block_sz(2): VLA_rd(2)
        VLA(:,:,count) = img(i: i+block_sz(1)-1, j: j+block_sz(2)-1);
        count = count +1;
    end
end

VLA_stress = VLA(:,:,1:10);
VLA_rest = VLA(:,:,11:20);

% Concatenate: ch.1~ch.40=Rest ; ch.41~ch.80=Stress
c1 = zeros(block_sz(1), block_sz(2), 80);
c1(:,:,1:20) = SA_rest;
c1(:,:,21:30) = HLA_rest;
c1(:,:,31:40) = VLA_rest;
c1(:,:,41:60) = SA_stress;
c1(:,:,61:70) = HLA_stress;
c1(:,:,71:80) = VLA_stress;

figure,
imagesc(c1(:,:,48))
colormap(gray(256))

%% Remove white bar at row 1 & col 1
c2 = c1(2:end,2:end,:);

figure,
imagesc(c2(:,:,48))
colormap(gray(256))

%% Remove number, method : apply nearest pixel value
p = [14, 17]; % processing area
w_threshold = 200;

tmp = c2(1:p(1), 1:p(2), :);
tmp = reshape(tmp, [], 1);
for i = 1: numel(tmp)
        while tmp(i) > w_threshold
            tmp(i) = tmp(i-1);
        end
end
tmp = reshape(tmp, p(1), p(2), []);

c3 = c2;
c3(1:p(1), 1:p(2), :) = tmp;

figure,
imagesc(c3(:,:,48))
colormap(gray(256))

%% Clinical Prior: (Rest - Stress) for SA, HLA, VLA
SA_diff = c3(:,:,1:20) - c3(:,:,41:60);
HLA_diff = c3(:,:,21:30) - c3(:,:,61:70);
VLA_diff = c3(:,:,31:40) - c3(:,:,71:80);
c4 = cat(3, SA_diff, HLA_diff, VLA_diff);

%% Maximum intensity projection (MIP)
% c5: ch1=SA MIP, ch2=HLA MIP, ch3=VLA MIP
c5 = zeros(size(c4,1), size(c4,2), 3);
c5(:,:,1) = max(c4(:,:,1:20),[],3); % SA MIP
c5(:,:,2) = max(c4(:,:,21:30),[],3); % HLA MIP
c5(:,:,3) = max(c4(:,:,31:40),[],3); % VLA MIP

figure, image(c5(:,:,1))
colormap(gray(256))
