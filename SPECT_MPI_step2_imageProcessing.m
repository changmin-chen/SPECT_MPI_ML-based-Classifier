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
    data = SPECT_MPI_imgProcFunc_ver1(img);
    niftiwrite(data, fullfile(targ, info(i).name)); % save as nifti
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
    data = SPECT_MPI_imgProcFunc_ver1(img);
    niftiwrite(data, fullfile(targ, info(i).name)); % save as nifti
end
