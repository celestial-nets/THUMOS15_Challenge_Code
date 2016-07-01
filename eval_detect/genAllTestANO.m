clear all;close all;clc

addpath('./data/');

load TEST_ANO
load score0524_W;
load TEST_FrameRate
load CONFID4
g_score_W = gpuArray(score0524_W);

pry_path = '/media/yuanjun/SeagateIV/pry5Frame/';
% obj_path = '/media/yuanjun/SeagateIV/TEST_OBJ_POOL_5FRAME/';


[ST_SC, ST_IDX] = sort(CONFID4, 2, 'descend');
T1 = ST_IDX(:, 1:1);
T3 = ST_IDX(:, 1:3);
T5 = ST_IDX(:, 1:5);


%%
ANO = cell(0);
for i = 1:size(uniqVid, 1)
    
    pry_name = [uniqVid{i, 1}, '.pry5Frame.mat'];
%     obj_name = [uniqVid{i, 1}, '.obj5Frame.mat'];
    
    if ~exist([pry_path, pry_name])
        continue;
    end
    
%     if ~exist([obj_path, obj_name])
%         continue;
%     end  
    
    load([pry_path, pry_name]); % SC, N, Frm
%     load([obj_path, obj_name]); % OBJ
    
    SC = gpuArray(SC);
    [FRM_SC, N_FEAT] = getFrmLBL(SC, N, g_score_W);

%     OBJ = gpuArray(OBJ);
%     [FRM_SC, N_FEAT] = genFrmLBL(SC, N, OBJ_FEAT, g_svmW);

    FRM_SC = gather(FRM_SC);
    
    ANO{i} = getANO(uniqVid{i, 1}, FRM_SC, N_FEAT, FrameRate(i), T1(i, :));
        
    disp(['Video ', num2str(i), ' of ', num2str(size(uniqVid, 1)), ' has finished.']);
    
end
ANO = cat(1, ANO{:});

% for i = 1:size(ANO, 1)
%     ANO{i, 1} = ANO{i, 1}(1:end-4);
% end

fileID = fopen('Run41.txt', 'w');

formatSpec = '%s %1.7f %1.7f %d %1.7f\n';
[nrows,ncols] = size(ANO);
for row = 1:nrows
    fprintf(fileID, formatSpec, ANO{row,:});
end

fclose(fileID);











