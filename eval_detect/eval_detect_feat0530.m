clear all;close all;clc

addpath(genpath('/home/yuanjun/Desktop/cvlibs/liblinear-1.96'));
load('./data/VAL_TM_FEAT0529.mat');

FEAT = cell(0);
N = cell(0);
LBL = cell(0);
for i = 1:size(VAL_TM_FEAT, 1)
    
    T = VAL_TM_FEAT{i, 2}';
    
    T = bsxfun(@rdivide, T, sqrt(sum(T .^2, 2)));
    
    FEAT{i} = T;
    N{i} = VAL_TM_FEAT{i, 3}';
    LBL{i} = VAL_TM_FEAT{i, 4}';
    
end
FEAT = cat(1, FEAT{:});
N = cat(1, N{:});
LBL = cat(1, LBL{:});

%% Filter small 'N's
FEAT( N < 250, : ) = [];
LBL ( N < 250 ) = [];
N ( N < 250 ) = [];

load FEAT_OBJSC
FEAT_OBJSC = bsxfun(@rdivide, FEAT_OBJSC, sqrt(sum(FEAT_OBJSC .^ 2, 2)));

FEAT = [FEAT, FEAT_OBJSC];

%%
DET = [7, 9, 12, 21, 22, ...
      23, 24, 26, 31, 33, ...
      36, 40, 45, 51, 68, ...
      79, 85, 92, 93, 97];
  
LBL2 = zeros(size(LBL));  
for i = 1:length(DET)
    LBL2(LBL == DET(i)) = i+1;
end
LBL2(LBL == 102) = 1;  
  
%%
str = [' -c 200 -w1 10'];
negCase = sum(LBL2 == 1);
for i = 2:21
    str = [str, ' -w', num2str(i), ' ', num2str(negCase/sum(LBL2 == i))];
end
str = [str, ' -B 1'];

%%
FEAT2_tr = cell(0);
FEAT2_ts = cell(0);
LBL2_tr = cell(0);
LBL2_ts = cell(0);

for i = 1:21
    TMP = FEAT(LBL2 == i, :);
    nTMP = size(TMP, 1);
    FEAT2_tr{i} = TMP(1 : floor(nTMP*2/3), :);
    FEAT2_ts{i} = TMP(floor(nTMP*2/3) + 1 : end, :);
    
    LBL2_tr{i} = ones(size(FEAT2_tr{i}, 1), 1) * i;
    LBL2_ts{i} = ones(size(FEAT2_ts{i}, 1), 1) * i;
end
FEAT2_tr{1} = FEAT2_tr{1}(1:15:end, :);
FEAT2_ts{1} = FEAT2_ts{1}(1:15:end, :);
LBL2_tr{1} = LBL2_tr{1}(1:15:end, :);
LBL2_ts{1} = LBL2_ts{1}(1:15:end, :);


FEAT2_tr = cat(1, FEAT2_tr{:});
FEAT2_ts = cat(1, FEAT2_ts{:});
LBL2_tr = cat(1, LBL2_tr{:});
LBL2_ts = cat(1, LBL2_ts{:});

%%
disp(['SVM ...']);

% FEAT2_tr1 = zeros(size(FEAT2_tr), 'single');
% FEAT2_ts1 = zeros(size(FEAT2_ts), 'single');
% for i = 1:101
%     FEAT2_tr1(:, (i*9-8) : (i*9) ) = sort(FEAT2_tr(:, (i*9-8) : (i*9) ), 2, 'descend');
%     FEAT2_ts1(:, (i*9-8) : (i*9) ) = sort(FEAT2_ts(:, (i*9-8) : (i*9) ), 2, 'descend');
% end
% 
% 
% FEAT2_tr1 = zeros(size(FEAT2_tr), 'single');
% FEAT2_ts1 = zeros(size(FEAT2_ts), 'single');
% for i = 1:101
%     
%     T1 = FEAT2_tr(:, (i*9-8) : (i*9) );
%     M  = mean(T1(:));
%     K  = std(T1(:));
%     
%     FEAT2_tr1(:, (i*9-8) : (i*9) ) = ( FEAT2_tr(:, (i*9-8) : (i*9) ) - M ) / K;
%     FEAT2_ts1(:, (i*9-8) : (i*9) ) = ( FEAT2_ts(:, (i*9-8) : (i*9) ) - M ) / K;
%     i
%     
% end
% 
% FEAT2_tr1 = bsxfun(@rdivide, FEAT2_tr1, sqrt(sum(FEAT2_tr1 .^2, 2)));
% FEAT2_ts1 = bsxfun(@rdivide, FEAT2_ts1, sqrt(sum(FEAT2_ts1 .^2, 2)));


mixSVM = train(LBL2_tr, sparse(double([FEAT2_tr])), str);
[plbl, pacc, pval] = predict(LBL2_ts, sparse(double([FEAT2_ts])), mixSVM);






