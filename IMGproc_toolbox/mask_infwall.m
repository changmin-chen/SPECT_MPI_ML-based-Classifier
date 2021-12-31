function [img_masked, mask] = mask_infwall(varargin)
% objective :
% get half-circle-mask or lower-linebar-mask for each blocks
% to mask out the possible(not always the case) excessive signal at inferior wall (come from bowels)
% -----
% mask = get_mask(img, 'mirror'), or
% mask = get_mask(img)
% -----
% mirror: 
% mirroring the mask so the mask for both the stress and rest blocks have the same shapes
% -----
% related function: get_centroids

img = varargin{1};

% transformation to the mask
if nargin > 1
    flag = varargin{2};
    
    if strcmp(flag, 'mirror')
        transform = @mask_mirroring;
    end
    
end

%% Get centroid and estimated left ventricle walls
[centroids, lvc, ~] = get_centroids(img);
stress_centroids = centroids(1:40, :);
rest_centroids = centroids(41:80, :);

%% Computing mask based on centroids
mask = true(size(lvc));
overlap_threshold = 0.75; % overlapping b/w area and circle-mask > 0.75
yaxis_threshold = 89/2; % distance to upper edge > distance to lower edge

% stress
count = 1;
for i = 1:2:7
    for j = 1:10
        lvc_p = lvc((i-1)*89+1: i*89, (j-1)*89+1: j*89);
        radius = sqrt(sum(sum(lvc_p))/pi);
        tmp_mask = draw_circle(...
            stress_centroids(count,2)...
            , stress_centroids(count,1)...
            , radius);
        if (sum(sum((lvc_p & tmp_mask))) /  sum(sum(lvc_p)) > overlap_threshold) && (stress_centroids(count,1)>=yaxis_threshold)
            tmp_mask(1:stress_centroids(count,1), :) = true; % we only need lower-half circle mask
            mask((i-1)*89+1: i*89, (j-1)*89+1: j*89) = tmp_mask;
        elseif (sum(sum((lvc_p & tmp_mask))) /  sum(sum(lvc_p)) <= overlap_threshold) && find(sum(lvc_p, 2), 1,'last') >= 80
            mask((i-1)*89+1: i*89, (j-1)*89+1: j*89) = [true(74, 89); false(15, 89)];
        end
        count = count+1;
    end
end
% rest
count = 1;
for i = 2:2:8
    for j = 1:10
        lvc_p = lvc((i-1)*89+1: i*89, (j-1)*89+1: j*89);
        radius = sqrt(sum(sum(lvc_p))/pi);
        tmp_mask = draw_circle(...
            rest_centroids(count,2)...
            , rest_centroids(count,1)...
            , radius);
        if (sum(sum((lvc_p & tmp_mask))) /  sum(sum(lvc_p)) >overlap_threshold) && (rest_centroids(count,1)>=yaxis_threshold)
            tmp_mask(1:rest_centroids(count,1), :) = true; % we only need lower-half circle mask
            mask((i-1)*89+1: i*89, (j-1)*89+1: j*89) = tmp_mask;
        elseif (sum(sum((lvc_p & tmp_mask))) /  sum(sum(lvc_p)) <= overlap_threshold) && find(sum(lvc_p, 2), 1,'last') >= 80
            mask((i-1)*89+1: i*89, (j-1)*89+1: j*89) = [true(74, 89); false(15, 89)];
        end
        count = count+1;
    end
end

%% Try to mirroring the mask if specified
try
mask = transform(mask);
end

%% Appling mask to the image
img_masked = img;
img_masked(~repmat(mask,[1,1,3])) = 0;

end

%---------------------------------------------
% helper func 1
%---------------------------------------------
function circle = draw_circle(x_center, y_center, radius)
% mask for a block element
[x, y] = meshgrid(1:89, 1:89);
distance = sqrt((x-x_center).^2 + (y-y_center).^2);
% draw circle mask
circle = true(89, 89);
circle(distance>radius) = false;

end

%---------------------------------------------
% helper func 2
%---------------------------------------------
function mask_out = mask_mirroring(mask_in)
% objective:
% mirroring the mask so the mask for both the stress and rest blocks have the same shapes
% -----
% input: unsymmetric mask 
% i.e. mask for rest block may be different from mask for stress block at the same slice
% -----
% output: mirrored mask
% mask for rest block would be same for mask for stress block.
% *mask is in logical format. ROI we want = True, ROI we don't want = False

mask_out = true(size(mask_in));

% column (left-right) neighbors mirroring
mask_in = ~mask_in;
for r = 1:8
    for c = 1:10
        valid = c-1>=1 && c+1<=10;
        if valid
            center =  mask_in((r-1)*89+1:r*89, (c-1)*89+1:c*89);
            lt_nei = mask_in((r-1)*89+1:r*89, (c-2)*89+1:(c-1)*89);
            rt_nei = mask_in((r-1)*89+1:r*89, c*89+1:(c+1)*89);
            mixed = (lt_nei | center) | rt_nei;         
            
            mask_out((r-1)*89+1:r*89, (c-1)*89+1:c*89) = mixed;
        elseif c-1 < 1 % left-most column
            center =  mask_in((r-1)*89+1:r*89, (c-1)*89+1:c*89);
            rt_nei = mask_in((r-1)*89+1:r*89, c*89+1:(c+1)*89);
            mixed = center | rt_nei;
            
            mask_out((r-1)*89+1:r*89, (c-1)*89+1:c*89) = mixed;
        elseif c+1 > 10 % right-most column
            center =  mask_in((r-1)*89+1:r*89, (c-1)*89+1:c*89);
            lt_nei = mask_in((r-1)*89+1:r*89, (c-2)*89+1:(c-1)*89);
            mixed = center | lt_nei;
            
            mask_out((r-1)*89+1:r*89, (c-1)*89+1:c*89) = mixed;
        end
    end
end

% row (up-down) neighbors mirroring
for r = 1: 2: 8
    st = mask_out((r-1)*89+1: r*89, :);
    rt = mask_out(r*89+1: (r+1)*89, :);
    mixed = st | rt;
    mask_out((r-1)*89+1: (r+1)*89, :) = repmat(mixed, [2,1]);
end

mask_out = ~mask_out;

end
