clear, clc
addpath('./helperFunctions')

% select image processing version
ver = 'ver0';
% ver = 'ver1';
% ver = 'ver2';

%% Perform image processing

% directory to save TrainSet
src_train = '../data/TrainSet';
targ_train = '../proc_data/TrainSet';
if ~exist(targ_train, 'dir')
    mkdir(targ_train)
end

% directory to save TestSet
src_test = '../data/TestSet';
targ_test = '../proc_data/TestSet';
if ~exist(targ_test, 'dir')
    mkdir(targ_test)
end

% do image processing
switch ver
    case 'ver0'
        fun = @SPECT_MPI_imgProcFunc_ver0;
    case 'ver1'
        fun = @SPECT_MPI_imgProcFunc_ver1;
    case 'ver2' 
        fun = @SPECT_MPI_imgProcFunc_ver2;
end

fprintf('Perform image processing for TrainSet\n')
do_IMGproc(src_train, targ_train, fun);
fprintf('done.\n')

fprintf('Perform image processing for TestSet\n')
do_IMGproc(src_test, targ_test, fun);
fprintf('done.\n')

%% helper functions
function do_IMGproc(src, targ, fun)

info = dir(fullfile(src,'*.jpg'));
for i = 1: length(info)
    img = imread(fullfile(info(i).folder, info(i).name));
    data = fun(img);
    niftiwrite(data, fullfile(targ, info(i).name));   % save as nifti
    disp(['Image processing for sample No.' num2str(i) ' completed.'])
end
end
