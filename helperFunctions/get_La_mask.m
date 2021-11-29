function I_close = get_La_mask(ccimg)
% this function will mirror the mask, see mask_mirroring
% input: RGB ccimg
% adjustable parameter: area_threshold
% larger the area_threshold, larger regions would be masked
% area_threshold = 150;
area_threshold = 250;
% area_threshold = 350;

% step 1: red channel threshod
idx = ccimg(:,:,1) <= 100;
ccimg(repmat(idx,[1,1,3])) = 0;

% step 2: Laplacian
kernal = [-1 -1 -1; -1 8 -1; -1 -1 -1];
L = double(rgb2gray(ccimg));
L = conv2(L, kernal, 'same');
Le = L>70; % edge

% step 3:  region growing
I = ~to3d(Le);
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
I_close = ~logical(toccimg(I)); 


end