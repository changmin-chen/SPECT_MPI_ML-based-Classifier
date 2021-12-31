function [centroids, lvc_wall, npx_list] = get_centroids(img, varargin)
% objective : get centroids of each blocks
% -----
% this function will optimize the threshold for heart wall
% threshold optimization rules:
% by defalut: threshold = 165, but
% reduce to 20 if heart wall is too small
% extend to 200 if heart wall is too large
% -----
% input:
% centroids = get_centroids(img, lower_limit, upper_limit)
% lower_limit & upper limit: wall pixel number threshold
% by default: 
% lower_limit = 500, upper_limit = 1800.
% -----
% output:
% centroids: the [row, col] coordinates of the centroids for each block
% lvc_wall: binary image representating the heart wall.
% npx_list: number of pixels used to estimate the centroids
% i.e. centroids = [row, col, slice number, npx]

%% Setting parameters
% use default limits if not specified
if nargin >1
    lower_limit = varargin{1};
    upper_limit = varargin{2};
else
    lower_limit = 500;
    upper_limit = 1800;
end

% coordinates for centroid calculation
[x, y] = meshgrid(1:89, 1:89); 

%% Get heart wall using red channel in rgb
R = to3d(img(:,:,1));

%% Finding the best threshold value for heart wall
wall_threshold = 165; % the default threshold (may be adjusted, see below)
centroids = zeros(size(R,3), 2); % save the [row, col] coordinates of centroids
npx_list = zeros(size(R,3), 1); % save the number of pixels used to estimate centroids
lvc_wall = false(size(R));
for p = 1: size(R, 3)
    Rp = R(:,:,p);
      
    % optimizing the threshold for the block element
    npx_pre = sum(sum(Rp>wall_threshold));
    if npx_pre < lower_limit
        threshold = 20;  % adjusted threshold
    elseif npx_pre > upper_limit
        threshold = 200; % adjusted threshold
    else
        threshold = wall_threshold;
    end
  
    % move to the next slice if the optimization failed
    npx_post = sum(sum(Rp>threshold));
    if (npx_post < lower_limit) || (npx_post > upper_limit)
        npx_list(p) = NaN;
    else    
        % otherwise, compute the centroid as the mass center of the heart wall
        npx_list(p) = npx_post;
        lvc_wall(:,:,p) = Rp >threshold;
        centroids(p, 1:2) = [mean(y(lvc_wall(:,:,p))), mean(x(lvc_wall(:,:,p)))];
    end
end

%% Assign the missed-block centroids as the whole-world centroid
available_centroids = centroids(~isnan(npx_list), :);
ww_centroid = mean(available_centroids);
centroids(isnan(npx_list), :) = repmat(ww_centroid, sum(isnan(npx_list)), 1);

% round the coordinates for indexing purpose
centroids = round(centroids);

% change the heart wall image from 3d format to 2d ccimg
lvc_wall = toccimg(lvc_wall);

end
