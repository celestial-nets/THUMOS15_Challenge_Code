clear all;close all;clc
% generate score - label features for validation set


addpath('./data');

load TM_ANO
DET = [7, 9, 12, 21, 22, ...
      23, 24, 26, 31, 33, ...
      36, 40, 45, 51, 68, ...
      79, 85, 92, 93, 97];
  
for i = 2:21
    for j = 1:size(TM_ANO{i}, 1)
        TM_ANO{i}{j, 9} = DET(i-1);
    end
end
T2 = cat(1, TM_ANO{2:end});
vidSeq = cat(1, T2{:, 7});
uniqVidSeq = unique(vidSeq);


  
VAL_TM_FEAT = cell(0);
for i = 1:length(uniqVidSeq) % catag
    
    idx = find(vidSeq == uniqVidSeq(i));
    vidName = ['thumos15_video_validation_', sprintf('%7.7d', uniqVidSeq(i)), '.mp4.pry5Frame.mat'];
%     load(['/home/yuanjun/Desktop/fvPool_5Frame_PRY_val/pry_val0529/', vidName]);
    load(['/home/yuanjun/Desktop/THUMOS_VAL_201507/thumos15_20150824/', vidName]);
    
    TMP = cell(0);
    for j = 1:size(SC, 3)
        TMP{j} = SC(:, :, j);
    end
    SC_FEAT = cat(1, TMP{:});
    N_FEAT  = N(1, :);
    
    LBL = 102 * ones(size(N_FEAT));
    for j = idx'
        
        FrmIntvStart = max(ceil(  (T2{j, 5} - 11) / 5 + 1), 1);
        FrmIntvEnd   = floor( (T2{j, 6} - 11) / 5 + 1);

        vidCatag = T2{j, 9};
        if ~any(vidCatag == DET)
            disp(['Error = ', num2str(i)]);
        end
    
        for k = FrmIntvStart : FrmIntvEnd
            if k > length(LBL)
                LBL(k) = vidCatag;
            elseif LBL(k) == 102
                LBL(k) = vidCatag;
            end
        end
        
    end
    
    VAL_TM_FEAT{i, 1} = vidName;
    VAL_TM_FEAT{i, 2} = SC_FEAT;
    VAL_TM_FEAT{i, 3} = N_FEAT;
    VAL_TM_FEAT{i, 4} = LBL;
    
    
    disp(['Video ', num2str(i), ' of ', num2str(length(uniqVidSeq)), ' has finished.']);
    
end

%%
for i = 1:size(VAL_TM_FEAT, 1)
    
    delta = length(VAL_TM_FEAT{i, 3}) - length(VAL_TM_FEAT{i, 4});
    if  delta ~= 0
        disp(['Delta = ', num2str(delta), ' VideoSeq ', num2str(i)]);
    end
    
end

ErrList = [166, 205, 228, 336, 384];
VAL_TM_FEAT(ErrList, :) = [];

for i = 1:size(VAL_TM_FEAT, 1)
    
    delta = length(VAL_TM_FEAT{i, 3}) - length(VAL_TM_FEAT{i, 4});
    if  delta ~= 0
        VAL_TM_FEAT{i, 4}(length(VAL_TM_FEAT{i, 3})+1 : end) = [];
        disp(['Delta = ', num2str(delta), ' VideoSeq ', num2str(i)]);
    end
    
end

save VAL_TM_FEAT0826 VAL_TM_FEAT -v7.3










