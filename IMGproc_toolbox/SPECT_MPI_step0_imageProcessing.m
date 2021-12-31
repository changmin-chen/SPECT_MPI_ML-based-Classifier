clear, clc

% directory to save TrainSet
src_train = '../../data/TrainSet';
targ_train = '../../proc_data/TrainSet';
if ~exist(targ_train, 'dir')
    mkdir(targ_train)
end

% directory to save TestSet
src_test = '../../data/TestSet';
targ_test = '../../proc_data/TestSet';
if ~exist(targ_test, 'dir')
    mkdir(targ_test)
end

%%  Select image processing version
version_list = {'ver0', 'ver1', 'ver2'};
subjstr = [{'{ver 0}: concatenate each blocks to 3D, then do nothing further.'},...
    {'{ver 1}: after the ver0 process, also perform coregistration and masking.'},...
    {'{ver 2}: after the ver0 process, also perform coregistation.'}];
subjsel = listdlg('liststring',subjstr,...
    'promptstring',[],...
    'selectionmode','single',...
    'name','Please Select the Image Processing Version',...
    'Listsize',[500 160]);
if isempty(subjsel), return, end

ver = version_list{subjsel};

%% Perform image processing
fprintf('Perform image processing for TrainSet\n')
do_IMGproc(src_train, targ_train, ver);
fprintf('done.\n')

fprintf('Perform image processing for TestSet\n')
do_IMGproc(src_test, targ_test, ver);
fprintf('done.\n')

%% helper functions
function do_IMGproc(src, targ, version)

info = dir(fullfile(src,'*.jpg'));
for i = 1: length(info)
    % read and process image
    img = imread(fullfile(info(i).folder, info(i).name));
    data = SPECT_MPI_imgProcFunc(img, version);
    
    % save as nifti
    [~, filename, ~] = fileparts(info(i).name);
    niftiwrite(data, fullfile(targ, [filename '.nii']));   
    disp(['Image processing for sample No.' num2str(i) ' completed.'])
end
end
