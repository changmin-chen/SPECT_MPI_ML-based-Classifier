clear, clc
close all

%% For TrainSet
src = '../data/TrainSet';
targ = '../proc_data/TrainSet';
if ~exist(targ, 'dir')
    mkdir(targ)
end

info = dir(fullfile(src,'*.jpg'));
for i = 1: length(info)
    img = imread(fullfile(info(i).folder, info(i).name));
    
    % please select image procesing function here, and comment the others
%     data = SPECT_MPI_imgProcFunc_ver0(img);
%     data = SPECT_MPI_imgProcFunc_ver1(img);
%     data = SPECT_MPI_imgProcFunc_ver2(img);
    data = SPECT_MPI_imgProcFunc_ver3(img);
    
    % save as nifti
    niftiwrite(data, fullfile(targ, info(i).name));
    disp(['Image processing for volume No.' num2str(i) ' completed.'])
end

%% For TestSet
src = '../data/TestSet';
targ = '../proc_data/TestSet';
if ~exist(targ, 'dir')
    mkdir(targ)
end

info = dir(fullfile(src,'*.jpg'));
for i = 1: length(info)
    img = imread(fullfile(info(i).folder, info(i).name));
    
    % please select image procesing function here, and comment the others
%     data = SPECT_MPI_imgProcFunc_ver0(img);
%     data = SPECT_MPI_imgProcFunc_ver1(img);
%     data = SPECT_MPI_imgProcFunc_ver2(img);
    data = SPECT_MPI_imgProcFunc_ver3(img);
    
     % save as nifti
    niftiwrite(data, fullfile(targ, info(i).name));
    disp(['Image processing for volume No.' num2str(i) ' completed.'])
end
