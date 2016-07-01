function [SC, N, Frm] = getPry4(fv_unorm, meta, W, sqrtPRI)

SC = cell(9, 1);
N = cell(9, 1);
Nm = size(fv_unorm, 2)-1;


TMP_FV = zeros([size(fv_unorm, 1), Nm], 'single', 'gpuArray');
PAD_N = padarray(meta(2,:), [0, 8]);
TMP_N = zeros(1, Nm, 'single');

for p = 1:9
  
        if p == 1
            TMP_FV= fv_unorm(:, 1:end-1) + fv_unorm(:, 2:end);
            TMP_N = meta(2, 1:end-1) + meta(2, 2:end);
	    fv_unorm = padarray(fv_unorm, [0, 8]);
        else
            TMP_FV = TMP_FV + fv_unorm(:, (1-p+1+8):(Nm-p+1+8)) + fv_unorm(:, (1+p+8):(Nm+p+8));
            TMP_N = TMP_N + PAD_N((1-p+1+8):(Nm-p+1+8)) + PAD_N((1+p+8):(Nm+p+8));
        end
        
    
    FV_N = normFV(TMP_FV', sqrtPRI);    
    FV_N = sign(FV_N) .* sqrt(abs(FV_N));
    FV_N = fv_l2norm(FV_N);
        
    SC{p} = gather(reshape(FV_N * W, 1, Nm, 101));
    N{p} = TMP_N;
    
    disp(['Pyramid: ', num2str(p), ' has finished.']);
    
end
SC = cat(1, SC{:});
N = cat(1, N{:});

Frm = 11:5:(11+(size(SC, 2)-1)*5);

end

function nfv = normFV(fv, sqrtPRI)

    nfv = bsxfun(@rdivide, fv, sqrtPRI);
    nfv = sign(nfv) .* sqrt(abs(nfv));

    v{1} = fv_l2norm(nfv(:, 1    :12288));
    v{2} = fv_l2norm(nfv(:, 12289:24576));

    v{3} = fv_l2norm(nfv(:, 24577:38400));
    v{4} = fv_l2norm(nfv(:, 38401:52224));

    v{5} = fv_l2norm(nfv(:, 52225:64512));
    v{6} = fv_l2norm(nfv(:, 64513:76800));

    v{7} = fv_l2norm(nfv(:, 76801:89088));
    v{8} = fv_l2norm(nfv(:, 89089:101376));
    
    nfv = cat(2, v{:});

end

function Y = fv_l2norm(X)

    Y = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2)));
    
end


% function nfv = normFV(fv, UCF_GMM)
% % {10240,24576,27648,24576,24576};
% 
%     v{1} = procNorm(fv(1    :12288), UCF_GMM.HOG.PRI);
%     v{2} = procNorm(fv(12289:24576), UCF_GMM.HOG.PRI);
% 
%     v{3} = procNorm(fv(24577:38400), UCF_GMM.HOF.PRI);
%     v{4} = procNorm(fv(38401:52224), UCF_GMM.HOF.PRI);
% 
%     v{5} = procNorm(fv(52225:64512), UCF_GMM.MBHX.PRI);
%     v{6} = procNorm(fv(64513:76800), UCF_GMM.MBHX.PRI);
% 
%     v{7} = procNorm(fv(76801:89088), UCF_GMM.MBHY.PRI);
%     v{8} = procNorm(fv(89089:101376), UCF_GMM.MBHY.PRI);
% 
%     nfv = cat(2, v{:});
% 
% end
% 
% function v = procNorm(fv_seg, pri)
% 
%     if ~any(fv_seg(:))
%         
%         v = zeros(1, numel(fv_seg), 'single');
%         
%     else
% 
%         M = reshape(fv_seg, [], 256);
% 
%         M = bsxfun(@rdivide, M, sqrt(pri));
%         M = sign(M) .* sqrt(abs(M));
% 
%         v = M(:)' / sqrt(sum(M(:) .^ 2));
%         
%     end
%     
% end



    
    
    
