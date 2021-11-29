function [Vout, tforms] = regist_3d(Vin)
% input:
% V, matrix size 89x89x80, stress 1:40, rest 41:80
% fixed: stress, moved: rest
[optimizer, metric] = imregconfig('monomodal');
regist_type = 'rigid';
% regist_type = 'affine';


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

rest(:,:,1:20) = imwarp(rest(:,:,1:20), tform_SA, 'OutputView', imref3d([89, 89, 20]));
rest(:,:,21:30) = imwarp(rest(:,:,21:30), tform_HLA, 'OutputView', imref3d([89, 89, 10]));
rest(:,:,31:40) = imwarp(rest(:,:,31:40), tform_VLA, 'OutputView', imref3d([89, 89, 10]));

Vout = cat(3, stress, rest);

end