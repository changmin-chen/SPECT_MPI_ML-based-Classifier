clear, clc
close all

src = '../1003.jpg'; % abnormal
block_sz = [90, 90];

img = imread(src);
% figure,
% imagesc(img)

%% Processing
% SA view
SA_lu = [51, 70];
SA_rd = [411, 970] - block_sz;
SA = zeros(block_sz(1), block_sz(2), 3, 40);
count = 1;
for i = SA_lu(1): block_sz(1): SA_rd(1)
    for  j = SA_lu(2): block_sz(2): SA_rd(2)
        SA(:,:,:,count) = img(i: i+block_sz(1)-1, j: j+block_sz(2)-1,:);
        count = count +1;
    end
end
SA_stress = SA(:,:,:,[1:10, 21:30]);
SA_rest = SA(:,:,:,[11:20, 31:40]);


% HLA view
HLA_lu = [484, 70];
HLA_rd = [664, 970] - block_sz;
HLA = zeros(block_sz(1), block_sz(2), 3, 20);
count = 1;
for i = HLA_lu(1): block_sz(1): HLA_rd(1)
    for  j = HLA_lu(2): block_sz(2): HLA_rd(2)
        HLA(:,:,:,count) = img(i: i+block_sz(1)-1, j: j+block_sz(2)-1,:);
        count = count +1;
    end
end
HLA_stress = HLA(:,:,:,1:10);
HLA_rest = HLA(:,:,:,11:20);

% VLA view
VLA_lu = [708, 70];
VLA_rd = [888, 970] - block_sz;
VLA = zeros(block_sz(1), block_sz(2), 3, 20);
count = 1;
for i = VLA_lu(1): block_sz(1): VLA_rd(1)
    for  j = VLA_lu(2): block_sz(2): VLA_rd(2)
        VLA(:,:,:,count) = img(i: i+block_sz(1)-1, j: j+block_sz(2)-1,:);
        count = count +1;
    end
end
VLA_stress = VLA(:,:,:,1:10);
VLA_rest = VLA(:,:,:,11:20);

% Concatenate: ch.1~ch.40=Rest ; ch.41~ch.80=Stress
c1 = zeros(block_sz(1), block_sz(2), 3, 80);
c1(:,:,:,1:20) = SA_rest;
c1(:,:,:,21:30) = HLA_rest;
c1(:,:,:,31:40) = VLA_rest;
c1(:,:,:,41:60) = SA_stress;
c1(:,:,:,61:70) = HLA_stress;
c1(:,:,:,71:80) = VLA_stress;

% figure,
% imagesc(uint8(c1(:,:,:,48)))

%% Remove white bar at row 1 & col 1
c2 = c1(2:end,2:end,:,:);

% figure,
% imagesc(uint8(c2(:,:,:,48)))

%% Remove number, method : apply nearest pixel value
p = [14, 17]; % processing area
w_threshold = 150;

tmp = c2(1:p(1), 1:p(2), :, :); % size = [p(1), p(2), 3, 80]
tmp = permute(tmp, [1, 2, 4, 3]); % size = [p(1), p(2), 80, 3]
tmp = reshape(tmp, p(1)*p(2)*80, 3); % size = [N, 3];
for i = 1: size(tmp,1)
        while all(tmp(i, :) > w_threshold)
            try
                tmp(i, :) = tmp(i+ceil(rand(1)*42-21), :);
            catch
            end
        end
end
tmp = reshape(tmp,p(1),p(2),80,3);
tmp = permute(tmp, [1, 2, 4, 3]);
c3 = c2;
c3(1:p(1), 1:p(2), :, :) = tmp;

% figure,
% imagesc(uint8(c3(:,:,:,48)))

%% RGB2gray
% rgb2gray
c3 = uint8(c3);
c4 = zeros(size(c3,1), size(c3,2), size(c3,4));
for i = 1: size(c4,3)
    c4(:,:,i) = rgb2gray(c3(:,:,:,i));
end

% figure, image(c4(:,:,10)), colormap(gray(256))
% figure, image(c4(:,:,50)), colormap(gray(256))

%% Substraction (Rest - Stress)
c5 = zeros(size(c4,1), size(c4,2), 40);
c5(:,:,1:20) = c4(:,:,1:20) - c4(:,:,41:60);
c5(:,:,21:30) = c4(:,:,21:30) - c4(:,:,61:70);
c5(:,:,31:40) = c4(:,:,31:40) - c4(:,:,71:80);

slope = 1;
intercept = 500;
c5 = c5*slope + intercept;
c5 = uint16(c5);

%% Check
% f1 = figure('Position',[616,754,560,420]);
% a1 = axes('Parent', f1);
% for i = 1:40
%     imagesc(c5(:,:,i), 'Parent', a1)
%     colormap(a1, gray(256))
%     pause
% end
