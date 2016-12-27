clear all;close all;clc

load('../../data/VAL_P5TM_FEAT0308.mat');
load('../../data/Val_14.mat');
% load('/media/yuanjun/Programes/THUMOS/THUMOS14_EXP/data/scW0906_P14.mat');

FEAT = cell(0);
N = cell(0);
LBL = cell(0);
BFLAG = cell(0); % background flag
NM = cell(0);
for i = 1:size(VAL_P5TM_FEAT, 1)
    
    T = VAL_P5TM_FEAT{i, 2}';
    
    T = bsxfun(@rdivide, T, sqrt(sum(T .^2, 2)));
    
%     N{i} = VAL_P5TM_FEAT{i, 3}';
    LBL{i} = VAL_P5TM_FEAT{i, 4}';

%     thres = 0.25 * median(N{i});
%     BFLAG{i} = (N{i} < thres) | (any(isnan(T), 2));
    BFLAG{i} = any(isnan(T), 2);
    T(BFLAG{i}, :) = 0;
    FEAT{i} = T;
    
    NM{i} = VAL_P5TM_FEAT{i, 1};
    
end


%%
DET = [7, 9, 12, 21, 22, ...
      23, 24, 26, 31, 33, ...
      36, 40, 45, 51, 68, ...
      79, 85, 92, 93, 97];

for j = 1:length(LBL)
    LBL2 = zeros(size(LBL{j}));  
    for i = 1:length(DET)
        LBL2(LBL{j} == DET(i)) = i+1;
    end
    LBL2(LBL{j} == 102) = 1;
    LBL{j} = LBL2;
end

TM_VID_IND = false(size(VAL_P5TM_FEAT, 1), 1); % 1-in TrSet, 2-in TsSet
for i = 1:size(VAL_P5TM_FEAT, 1)
    idx = str2double(VAL_P5TM_FEAT{i, 1}(28:33));
    if id.tr_in_tm(idx) == true
        TM_VID_IND(i) = true;
    elseif id.ts_in_tm(idx) == true
        TM_VID_IND(i) = false;
    else
        error('');
    end
    
end

%%
% FEAT_TM = cell(0);
% for i = 1:length(FEAT)
%     FEAT_TM{i} = [FEAT{i}, ones(size(FEAT{i}, 1), 1)] * scW0906_P14;
% end
FEAT_TM = FEAT;

clearvars -except FEAT_TM TM_VID_IND LBL N BFLAG NM

%%
rng(0);

SEG_LEN = 20;
SEG_LEN1 = round(1.1*SEG_LEN);
FEAT_SEG = cell(0);
LBL_SEG = cell(0);
FLAG_SEG = cell(0);
MSK_SEG = cell(0);
IND_DROP = cell(0);

for i = 1:length(FEAT_TM)

    tmp_feat = FEAT_TM{i};
    tmp_lbl = LBL{i};
    
    nSeg = floor((size(FEAT_TM{i}, 1) - SEG_LEN/10) / SEG_LEN);
    for j = 1:nSeg
        FEAT_SEG{i}{j} = tmp_feat(1:SEG_LEN1, :);
        LBL_SEG{i}{j} = tmp_lbl(1:SEG_LEN1);
        MSK_SEG{i}{j} = true(size(LBL_SEG{i}{j}));
        
        if sum(tmp_lbl(1:SEG_LEN)) == SEG_LEN
            FLAG_SEG{i}{j} = 0;
            if rand > .2;
                IND_DROP{i}{j} = 1;
            else
                IND_DROP{i}{j} = 0;
            end
        else
            FLAG_SEG{i}{j} = 1;
            IND_DROP{i}{j} = 0;
        end
        
        tmp_feat(1:SEG_LEN, :) = [];
        tmp_lbl(1:SEG_LEN) = [];
        
    end
    
    if length(tmp_lbl) > SEG_LEN/4
        len = length(tmp_lbl);
        FEAT_SEG{i}{nSeg+1} = [tmp_feat(1:len, :); zeros( SEG_LEN1 - len, 909)];
        LBL_SEG{i}{nSeg+1} = [tmp_lbl(1:len); zeros(SEG_LEN1 - len, 1)];
        MSK_SEG{i}{nSeg+1} = (LBL_SEG{i}{nSeg+1} ~= 0);

        if sum(tmp_lbl) == length(tmp_lbl);
            FLAG_SEG{i}{nSeg+1} = 0;
            if rand > .2
                IND_DROP{i}{nSeg+1} = 1;
            else
                IND_DROP{i}{nSeg+1} = 0;
            end            
        else
            FLAG_SEG{i}{nSeg+1} = 1;
            IND_DROP{i}{nSeg+1} = 0;
        end
        
    end
    
    
end

clearvars -except FEAT_SEG LBL_SEG FLAG_SEG SEG_LEN MSK_SEG IND_DROP TM_VID_IND NM LBL

%%
trFEAT_SEG = FEAT_SEG(TM_VID_IND);
trLBL_SEG = LBL_SEG(TM_VID_IND);
trFLAG_SEG = FLAG_SEG(TM_VID_IND);
trMSK_SEG = MSK_SEG(TM_VID_IND);
trIND_DROP = IND_DROP(TM_VID_IND);

tsFEAT_SEG = FEAT_SEG(~TM_VID_IND);
tsLBL_SEG = LBL_SEG(~TM_VID_IND);
tsFLAG_SEG = FLAG_SEG(~TM_VID_IND);
tsMSK_SEG = MSK_SEG(~TM_VID_IND);
tsIND_DROP = IND_DROP(~TM_VID_IND);

trLBL = LBL(TM_VID_IND);
tsLBL = LBL(~TM_VID_IND);

ind = 1;
trFEAT_CAT = cell(0);
trLBL_CAT = cell(0);
trMSK_CAT= cell(0);
trLEN = cell(0);
for i = 1:length(trFEAT_SEG)
    for j = 1:length(trFEAT_SEG{i})
%         if trIND_DROP{i}{j} == 0
            trFEAT_CAT{ind} = trFEAT_SEG{i}{j};
            trLBL_CAT{ind} = trLBL_SEG{i}{j};
            trMSK_CAT{ind} = trMSK_SEG{i}{j};
            ind = ind + 1;
%         end
    end
    
end

ind = 1;
tsFEAT_CAT = cell(0);
tsLBL_CAT = cell(0);
tsMSK_CAT= cell(0);
for i = 1:length(tsFEAT_SEG)
    for j = 1:length(tsFEAT_SEG{i})
        tsFEAT_CAT{ind} = tsFEAT_SEG{i}{j};
        tsLBL_CAT{ind} = tsLBL_SEG{i}{j};
        tsMSK_CAT{ind} = tsMSK_SEG{i}{j};
        ind = ind + 1;
    end
    
end

clearvars -except trFEAT_CAT trLBL_CAT trMSK_CAT tsFEAT_CAT tsLBL_CAT tsMSK_CAT trLBL tsLBL

%%
wt_tr = zeros(1, 21);
for i = 1:length(trLBL_CAT)
    for j = 1:21
        wt_tr(j) = wt_tr(j) + sum(trLBL_CAT{i} == j);
    end
end
wt_tr = wt_tr / 1000;

wt_ts = zeros(1, 21);
for i = 1:length(tsLBL_CAT)
    for j = 1:21
        wt_ts(j) = wt_ts(j) + sum(tsLBL_CAT{i} == j);
    end
end
wt_ts = wt_ts / 1000;

clear i j

save P5D14_P22_PY_nodrop









