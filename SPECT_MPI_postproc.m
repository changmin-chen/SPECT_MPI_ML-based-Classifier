function fmap = SPECT_MPI_postproc(c)

%% postproc option 1
% % split into each class
% SA1_rest = r_shape(c(:,:,2:10)); % remove first
% SA2_rest = r_shape(c(:,:,11:19)); % remove last
% HLA_rest = r_shape(c(:,:,21:29)); % remove last
% VLA_rest = r_shape(c(:,:,31:39)); % remove last
% SA1_stress = r_shape(c(:,:,42:50)); % remove first
% SA2_stress = r_shape(c(:,:,51:59)); % remove last
% HLA_stress = r_shape(c(:,:,61:69)); % remove last
% VLA_stress = r_shape(c(:,:,71:79)); % remove last
% 
% % concatenate
% fmap_rest = [SA1_rest, HLA_rest; VLA_rest, SA2_rest]; % dtype: double
% fmap_stress = [SA1_stress, HLA_stress; VLA_stress, SA2_stress]; % dtype: double
% 
% % feature map
% fmap = cat(3, fmap_rest, fmap_stress); % rest, stress
% fmap = int8(fmap); % dtype: integer 8

%% postproc option 2
rest = c(:,:,1:40);
stress = c(:,:,41:80);
fmap = cat(4, rest ,stress);


% helper functions
function M = r_shape(V)
M = zeros(89*3, 89*3);
count = 1;
for i = 1:3
    for j = 1:3
       M((i-1)*89+1: i*89, (j-1)*89+1: j*89)...
           = V(:,:,count);
       count = count +1;
    end
end
end

end