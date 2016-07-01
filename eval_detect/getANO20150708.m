function ANO = getANO20150708(vidName, FRM_SC, N_FEAT, FRM_RATE, TOP)

DET = [102, 7, 9, 12, 21, 22, ...
      23, 24, 26, 31, 33, ...
      36, 40, 45, 51, 68, ...
      79, 85, 92, 93, 97];
  
%% Filter

[~, FRM_LBL] = max(FRM_SC, [], 1);

    
FRM_LBL(N_FEAT < 250) = 1;
% FRM_LBL_FILT = medfilt1(FRM_LBL, 11);
% FRM_LBL_FILT = medfilt1(FRM_LBL_FILT, 11);
% FRM_LBL_FILT = medfilt1(FRM_LBL_FILT, 11);
% FRM_LBL_FILT(FRM_LBL_FILT == 0) = 1;
FRM_LBL_FILT = FRM_LBL;

T = FRM_LBL_FILT;
BLK = cell(0);
ind = 1;

while 1
    
    idx = find( T ~= T(1), 1, 'first');
    if ~isempty(idx)
        BLK{1, ind} = T(1 : idx-1 );
        ind = ind + 1;
        
        T(1 : idx-1 ) = [];
        
    else
        break;
    end
end
BLK{1, ind} = T;

%% BLK Processing
ptr = 0;
for i = 1:size(BLK, 2)
    BLK{2, i} = ptr + [1:length(BLK{1, i})];
    ptr = ptr + length(BLK{1, i});
end

NON_BKG = cell(0);
ind = 1;
for i = 1:size(BLK, 2)

    if BLK{1, i}(1) == 1
        continue;
    else
        NON_BKG{1, ind} = BLK{1, i};
        NON_BKG{2, ind} = BLK{2, i};
        ind = ind + 1;
    end
    
end

numCase = size(NON_BKG, 2);
ANO = cell(numCase, 5);
for i = 1:numCase

    ANO{i, 1} = vidName;
    ANO{i, 2} = ((NON_BKG{2, i}(1)   - 1) * 5 + 11) / FRM_RATE;
    ANO{i, 3} = ((NON_BKG{2, i}(end) - 1) * 5 + 11) / FRM_RATE;
    ANO{i, 4} = DET(NON_BKG{1, i}(1));
    
%     SC_INTV = NON_BKG{2, i};
%     ANO{i, 5} = mean(FRM_SC(NON_BKG{1, i}(1), SC_INTV)) / 2 + .5;    
%     ANO{i, 5}(ANO{i, 5} < 0) = 0;
%     ANO{i, 5}(ANO{i, 5} > 1) = 1;
    ANO{i, 5} = 1;
    
end

idx = false(size(ANO, 1), 1);
for i = 1:size(ANO, 1)

    if ( (ANO{i, 3} - ANO{i, 2}) < .75 ) || (ANO{i, 5} == 0)        
%         idx(i) = true;        
    end
    
    if ~isempty(TOP)        
        if ~any( ANO{i, 4} == TOP )
            idx(i) = true;
        end        
    end
    
end
ANO(idx, :) = [];




    
    
    
    
    
    
end
