function data = SPECT_MPI_imgProcFunc_ver2(img)
% SPECT_MPI_imgProcFunc_ver 2

addpath('./helperFunctions');
img = cc_img(img); % ccimg size = 712x890x3

%% step 1: Get small region mask
I_close = get_small_region(img);
img_closed = img;
img_closed(repmat(I_close, [1,1,3])) = 0;

%% step 2: 3D-registration
% registration is estimated on closed images, however, registration is perfromed on original image
perf_img = to3d(rgb2gray(img_closed));
[~, tforms] = regist_3d(perf_img);

tmp = to3d(rgb2gray(img));
stress = tmp(:,:,1:40);
rest = tmp(:,:,41:80);
rest(:,:,1:20) = imwarp(rest(:,:,1:20), tforms{1}, 'OutputView', imref3d([89, 89, 20]));
rest(:,:,21:30) = imwarp(rest(:,:,21:30), tforms{2}, 'OutputView', imref3d([89, 89, 10]));
rest(:,:,31:40) = imwarp(rest(:,:,31:40), tforms{3}, 'OutputView', imref3d([89, 89, 10]));

%% output
% because I_close for stress and rest are same, we only save the one.
stress = uint8(stress);
rest = uint8(rest);
I_close = uint8(to3d(I_close));
data = cat(4, stress, rest, I_close(:,:,1:40)); 

end

