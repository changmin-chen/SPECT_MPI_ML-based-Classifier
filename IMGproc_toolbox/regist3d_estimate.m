function tforms = regist3d_estimate(Vin)
% input:
% V, matrix size 89x89x80,
% where stress = V(:,:,1:40), rest = V(:,:,41:80)
% -----
% output:
% estimated transformation matrices for SA, HLA and VLA view respectively
% e.g. tforms{1} = transformation matrix, for registering rest_SA to stress_SA
% for applying transformation matrix, see "regist3d_estimate_and_reslice"
% ---
% registration info. :
% fixed image: stress (i.e. reference image)
% moved image: rest
% type: rigid body

[optimizer, metric] = imregconfig('monomodal');
regist_type = 'rigid';

stress = Vin(:,:,1:40);
rest = Vin(:,:,41:80);

% estimate and perform 3D registration
tform_SA = imregtform(rest(:,:,1:20), stress(:,:,1:20),...
    regist_type, optimizer, metric);
tform_HLA = imregtform(pad_wall(rest(:,:,21:30)), pad_wall(stress(:,:,21:30)),...
    regist_type, optimizer,metric);
tform_VLA = imregtform(pad_wall(rest(:,:,31:40)), pad_wall(stress(:,:,31:40)),...
    regist_type, optimizer,metric);
tforms = {tform_SA, tform_HLA, tform_VLA}; % affine matrices

end

function P = pad_wall(W)
% helper func : padding for 3D registration
P = zeros(89, 89, 16);
if size(W, 3) == 10
    P(:,:,4:13) = W;
else
    P = W;
end
end