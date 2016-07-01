clear all;close all;clc

addpath('./data');
load TEST_ANO
load SVM_GFXYW
load UCF_GMM

pool_path = '/media/yuanjun/SeagateIV/pool5Frame/';
% pool_path = '/media/yuanjun/MyBook/pool_5Frame2/';
% pool_path = '/media/yuanjun/MyBook/pool_5Frame2/';

pry_path = '/home/yuanjun/Desktop/pry5Frame1/';

sqrtPRI = sqrt([kron(UCF_GMM.HOG.PRI, ones(1, 48)), kron(UCF_GMM.HOG.PRI, ones(1, 48)), ...
                kron(UCF_GMM.HOF.PRI, ones(1, 54)), kron(UCF_GMM.HOF.PRI, ones(1, 54)), ...
                kron(UCF_GMM.MBHX.PRI, ones(1, 48)), kron(UCF_GMM.MBHX.PRI, ones(1, 48)), ...
                kron(UCF_GMM.MBHY.PRI, ones(1, 48)), kron(UCF_GMM.MBHY.PRI, ones(1, 48))]);
            
g_sqrtPRI = gpuArray(sqrtPRI);
g_gfxyW = gpuArray(gfxyW);

%%
for i = 1:size(uniqVid, 1)

    pry_name = [uniqVid{i, 1}, '.pry5Frame.mat'];
    if exist([pry_path, pry_name]) %#ok<EXIST>
        continue;
    end
    
    pool_name = [uniqVid{i, 1}, '.pool5Frame.h5'];   
    if ~exist([pool_path, pool_name]) %#ok<EXIST>
        continue;
    end
    disp(['Video Sequence: ', num2str(i), ', Name: ', uniqVid{i, 1}, ', PRY Processing...']);

    tic;
    PL = h5read([pool_path, pool_name], '/d');
    load([pool_path, pool_name, '.mat']);
    Tm = toc;
    disp(['Loading Data Finished. Time = ', num2str(Tm), ', File Size: ', num2str(numel(PL)*4/1024^2), ' MB']);

    tic;
    g_PL = gpuArray(PL);
    [SC, N, Frm] = getPry_gpu(g_PL, meta_5Frame_unorm, g_gfxyW, g_sqrtPRI);
    Tm = toc;
    disp(['PRY Process Finished. Time = ', num2str(Tm)]);
    
    tic;
    save([pry_path, pry_name], 'SC', 'N', 'Frm');
    Tm = toc;
    disp(['Saving Finished. Time = ', num2str(Tm)]);
    
end
    
    
    
    
    
