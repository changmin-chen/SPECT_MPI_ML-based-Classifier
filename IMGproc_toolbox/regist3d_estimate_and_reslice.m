function rgb = regist3d_estimate_and_reslice(rgb)

% use grayscale 3d image to estimate the registration matrices
gray_img = rgb2gray(rgb);
tforms = regist3d_estimate(to3d(gray_img));

% apply the registration and reslice the image (same operation for R, G and B channel)
for ch = 1:3 % R, G and B
    % extract image channel by channel
    img_ch = to3d(rgb(:,:,ch));   
    stress = img_ch(:,:,1:40);
    rest = img_ch(:,:,41:80);
    
    % rest volume is registered to stress volume
    rest(:,:,1:20) = imwarp(rest(:,:,1:20), tforms{1}, 'OutputView', imref3d([89, 89, 20]));
    rest(:,:,21:30) = imwarp(rest(:,:,21:30), tforms{2}, 'OutputView', imref3d([89, 89, 10]));
    rest(:,:,31:40) = imwarp(rest(:,:,31:40), tforms{3}, 'OutputView', imref3d([89, 89, 10]));
    
    % rewrite the registered image to that channel
    img_ch_registed = cat(3, stress, rest);
    rgb(:,:, ch) = toccimg(img_ch_registed);
end

end