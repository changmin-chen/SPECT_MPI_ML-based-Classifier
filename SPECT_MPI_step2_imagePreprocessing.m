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
    img = SPECT_MPI_preproc(img);
    imwrite(img, fullfile(targ, info(i).name));
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
    img = SPECT_MPI_preproc(img);
    imwrite(img, fullfile(targ, info(i).name));
end

