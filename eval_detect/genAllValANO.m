clear all;close all;clc

addpath('./data');

load TM_ANO
load scW;
load VAL_FrameRate
g_scW = gpuArray(scW);

% pry_path = '/media/yuanjun/EAF0C6BCF0C68DEF/PRY_VAL_FEAT/';
pry_path = '/home/yuanjun/Desktop/THUMOS_VAL_201507/thumos15_20150824/';
% obj_path = '/media/yuanjun/EAF0C6BCF0C68DEF/thumos15_valfeat/OBJ_INTV_FEAT_5FRAME/';

%%
ANO = cell(0);
for i = 1:size(uniqVid, 1)
    
    pry_name = [uniqVid{i, 1}, '.mp4.pry5Frame.mat'];
%     obj_name = [uniqVid{i, 1}, '.mp4.obj5Frame.mat'];
    
    if ~exist([pry_path, pry_name])
        continue;
    end
    
%     if ~exist([obj_path, obj_name])
%         continue;
%     end 
    
    load([pry_path, pry_name]); % SC, N, Frm
%     load([obj_path, obj_name]); % OBJ
    
    SC = gpuArray(SC);
    [FRM_SC, N_FEAT] = getFrmLBL(SC, N, g_scW);

%     OBJ = gpuArray(OBJ);
%     [FRM_SC, N_FEAT] = genFrmLBL(SC, N, OBJ_FEAT, g_svmW);

    FRM_SC = gather(FRM_SC);
    
    ANO{i} = getANO([uniqVid{i, 1}, '.mp4'], FRM_SC, N_FEAT, FrameRate(i), []);
        
    disp(['Video Seq: ', num2str(i), ' of 413 has finished.']);
    
end
ANO = cat(1, ANO{:});

%% Filtering ANO - del small segs

fileID = fopen('val_run0826.txt', 'w');

formatSpec = '%s %1.7f %1.7f %d %1.7f\n';
[nrows,ncols] = size(ANO);
for row = 1:nrows
    fprintf(fileID, formatSpec, ANO{row,:});
end

fclose(fileID);










