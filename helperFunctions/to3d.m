function D = to3d(ccimg)
% helper func3: ccimg to 3D
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
% ordering, D(:,:,1:40) = stress, and D(:,:,41:80) = rest
stress_idx = [1:10, 21:30, 41:50, 61:70];
rest_idx = stress_idx +10;
D = D(:,:,[stress_idx, rest_idx]);

end