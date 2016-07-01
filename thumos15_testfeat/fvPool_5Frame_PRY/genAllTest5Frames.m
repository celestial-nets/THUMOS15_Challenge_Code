clear all;close all;clc

addpath('./data');
load UCF_PCA
load UCF_GMM
load TEST_ANO

addpath('~/Desktop/vlfeat-0.9.20/toolbox/');
vl_setup();

traj_path = '/media/DATA2/Projects/data/2015/TestData/testtraj_h5/';
pool_path = '/media/SSD/Projects/data/2015/TestData/pool5Frame/';

%%
for i = 1:size(uniqVid, 1)
    
    pool_name = [uniqVid{i, 1}, '.pool5Frame.h5'];
    if exist([pool_path, pool_name]) %#ok<EXIST>
        continue;
    end
    
    traj_name = [uniqVid{i, 1}, '.features.h5'];   
    if ~exist([traj_path, traj_name]) %#ok<EXIST>
        continue;
    end
    disp(['Video Sequence: ', num2str(i), ', Name: ', uniqVid{i, 1}, ', FV Processing...']);

    tic;
    TJ = h5read([traj_path, traj_name], '/d')';
    Tm = toc;
    disp(['Loading Data Finished. Time = ', num2str(Tm), ', File Size: ', num2str(numel(TJ)*4/1024^3), ' GB']);
    
    tic;
    [fv_5Frame_unorm, meta_5Frame_unorm] = getFV_5frame(TJ, uniqVid{i, 2}, UCF_PCA, UCF_GMM);
    Tm = toc;
    disp(['FV Process Finished. Time = ', num2str(Tm)]);
    
    tic;
    h5create([pool_path, pool_name], '/d', size(fv_5Frame_unorm), 'Datatype', 'single');
    h5write([pool_path, pool_name], '/d', fv_5Frame_unorm);
    save([pool_path, pool_name, '.mat'], 'meta_5Frame_unorm');
    Tm = toc;
    disp(['Saving Finished. Time = ', num2str(Tm)]);
    
end











