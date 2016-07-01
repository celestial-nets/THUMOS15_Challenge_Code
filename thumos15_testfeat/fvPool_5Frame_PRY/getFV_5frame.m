function [FV, META] = getFV_5frame(TJ, frmNum, UCF_PCA, UCF_GMM)

frameEndInd = TJ(:, 1);
feat = appPCA(TJ, UCF_PCA);
clear TJ;

FV = cell(0);
META = cell(0);

frmInd = 6;
ind = 1;
trjCnt = 0;
while 1

    if frmInd > ( frmNum - 9 )
        break;
    end
    % frame from 6 to end-9, ending frame from 6+7 to end-2
    
    frmStart = frmInd + 7;
    frmEnd = frmInd + 4 + 7;
    
    idx = ( (frameEndInd >= frmStart) & (frameEndInd <= frmEnd) );
    N = sum(idx);
    trjCnt = trjCnt + N;
    
    FV{ind} = getFisher_unorm(feat, idx, UCF_GMM);
    META{1, ind} = frmInd;
    META{2, ind} = N;

    if mod(ind, 100) == 0
        disp(['Frame ', num2str(frmInd), ' of ', num2str(frmNum), ' has finished.']);
    end

    frmInd = frmInd + 5;
    ind = ind + 1;

end

FV = cat(2, FV{:});
META = single(cell2mat(META));

end

function feat = appPCA(TJ, UCF_PCA)

    TJ = single(TJ);
    feat{1} = bsxfun(@minus, TJ(:, 41 :136), UCF_PCA.HOG.AVG)  * UCF_PCA.HOG.PROJ;
    feat{2} = bsxfun(@minus, TJ(:, 137:244), UCF_PCA.HOF.AVG)  * UCF_PCA.HOF.PROJ;
    feat{3} = bsxfun(@minus, TJ(:, 245:340), UCF_PCA.MBHX.AVG) * UCF_PCA.MBHX.PROJ;
    feat{4} = bsxfun(@minus, TJ(:, 341:436), UCF_PCA.MBHY.AVG) * UCF_PCA.MBHY.PROJ;
    
end

function fv = getFisher_unorm(feat, idx, UCF_GMM)

    hogfv  = vl_fisher_unorm(feat{1}(idx, :)', UCF_GMM.HOG.MEAN,  UCF_GMM.HOG.COV,  UCF_GMM.HOG.PRI);
    hoffv  = vl_fisher_unorm(feat{2}(idx, :)', UCF_GMM.HOF.MEAN,  UCF_GMM.HOF.COV,  UCF_GMM.HOF.PRI);
    mbhxfv = vl_fisher_unorm(feat{3}(idx, :)', UCF_GMM.MBHX.MEAN, UCF_GMM.MBHX.COV, UCF_GMM.MBHX.PRI);
    mbhyfv = vl_fisher_unorm(feat{4}(idx, :)', UCF_GMM.MBHY.MEAN, UCF_GMM.MBHY.COV, UCF_GMM.MBHY.PRI);

    fv = [hogfv; hoffv; mbhxfv; mbhyfv];

end

