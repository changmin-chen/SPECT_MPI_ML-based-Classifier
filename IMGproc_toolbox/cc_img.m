function C = cc_img(img)
% ***this function should be applied at first step*** 
% no matter which version of the image-processing protocol is performed.
% ----
% objective: 
% remove redudant things on the original image,
% then crop and concatenate into new 2d image
% original block element size = 90x90
% output block element size = 89x89 (discard the white block lines)
% ---
% input image size: 897x976x3
% output image size: 712x890x3

% SA view
SA_lu = [51, 70];
SA_rd = [411, 970];
SAr = setxor(SA_lu(1):SA_rd(1), SA_lu(1):90:SA_rd(1));
SAc = setxor(SA_lu(2):SA_rd(2), SA_lu(2):90:SA_rd(2));
% HLA view
HLA_lu = [484, 70];
HLA_rd = [664, 970];
HLAr = setxor(HLA_lu(1):HLA_rd(1), HLA_lu(1):90:HLA_rd(1));
HLAc = setxor(HLA_lu(2):HLA_rd(2), HLA_lu(2):90:HLA_rd(2));
% VLA view
VLA_lu = [708, 70];
VLA_rd = [888, 970];
VLAr = setxor(VLA_lu(1):VLA_rd(1), VLA_lu(1):90:VLA_rd(1));
VLAc = setxor(VLA_lu(2):VLA_rd(2), VLA_lu(2):90:VLA_rd(2));
% concatenate
SA = img(SAr,SAc,:);
HLA = img(HLAr,HLAc,:);
VLA = img(VLAr,VLAc,:);
C = cat(1, SA, HLA, VLA);

% remove number label on each block
proc_area = [14, 17];
for j = 1:10
    for i = 1:8
        C((i-1)*89+1: (i-1)*89+proc_area(1), (j-1)*89+1: (j-1)*89+proc_area(2), :) = 0;
    end
end

end