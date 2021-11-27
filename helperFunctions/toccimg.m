function ccimg = toccimg(D)
% helper func 4: 3D to ccimg
% 3D volume  size = 89x89x80, (stress:1:40, rest:41:80)
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