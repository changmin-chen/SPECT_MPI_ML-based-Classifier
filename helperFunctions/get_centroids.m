function centroids = get_centroids(wall, lower_limit, upper_limit)
% helper func 5: get centroids of each blocks
% threshold setting: 20 if too small, 165 is gold standard, 200 if excessive
% input:
% wall: red channel
% lower, upper limit: wall pixel number threshold
% SA: lower_limit = 500, upper_limit = 2300;
% HLA: lower_limit = 500, upper_limit = 2300;
% VLA:  lower_limit = 500, upper_limit = 2300;
% output:
% centroids: the table of centroids, with npx at the last column, where...
% npx: original number of pixels of each block element using default threshold
% i.e. centroids = [row, col, slice number, npx]
if any(size(wall)~=[89, 89, 20]) && any(size(wall)~=[89, 89, 10])
    error('input matrix size should be 89x89x20(SA) or 89x89x10(HLA,VLA). And please assign the rest and stress volumes separately.')
elseif isa(wall, 'logical')
    error('input should be Red-channel, not binary mask.')
end

default_threshold = 165; % the default threshold (may be adjusted, see below)
npx = zeros(size(wall,3), 1); % save the number of pixels of each block element
tmp = []; % temporarily save the centroids of part of the blocks
[x, y] = meshgrid(1:89, 1:89); % coordinates for centroid calculation

for p = 1: size(wall,3)
    pwall = wall(:,:,p);
    npx(p, 1) = sum(sum(pwall>default_threshold));
    
    % testing the better threshold for the block element
    if sum(sum(logical(pwall>default_threshold))) < lower_limit
        threshold = 20;  % adjusted threshold
    elseif sum(sum(logical(pwall>default_threshold))) > upper_limit
        threshold = 200; % adjusted threshold
    else
        threshold = default_threshold;
    end
    
    % wall size after threshold adjustion
    if sum(sum(pwall>threshold)) < lower_limit || sum(sum(pwall>threshold)) > upper_limit
        continue % move to the next plane if the wall size still can't meet the requirement
    else
        pwall = pwall>threshold;
        center_row = mean(y(pwall));
        center_col = mean(x(pwall));
        tmp = [tmp;...
            center_row, center_col, p];
    end
end
ww_centroid = mean(tmp(:,1:2), 1); % whole-world centroid
centroids = zeros(size(wall,3), 3); % save the final decided centroids of all blocks
centroids(tmp(:,3), :) = tmp; % assign the centroid for each block element if available

% assign the missed-block centroids as the whole-world centroid
miss_blocks = setxor(1:size(wall,3), tmp(:,3))';
for p = miss_blocks
    centroids(p, :) = [ww_centroid, p];
end
centroids = [centroids, npx];

end