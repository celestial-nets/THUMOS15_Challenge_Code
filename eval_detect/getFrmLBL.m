function [FRM_SC, N_FEAT] = getFrmLBL(SC, N, svmW)

%     if ( size(SC, 2) ~= size(N, 2) ) || ...
%        ( size(SC, 2) ~= size(OBJ_FEAT, 2) ) || ...
%        ( size(N,  2) ~= size(OBJ_FEAT, 2) )
%    
%         error('Dim mismatch');
%     end
        
    TMP = cell(0);
    for j = 1:size(SC, 3)
        TMP{j} = SC(:, :, j);
    end
    SC_FEAT = cat(1, TMP{:});
    N_FEAT  = N(1, :);

    SC_FEAT = SC_FEAT';
%     OBJ_FEAT = OBJ_FEAT';
    
    SC_FEAT = bsxfun(@rdivide, SC_FEAT, sqrt(sum(SC_FEAT .^2, 2)));    
%     OBJ_FEAT = bsxfun(@rdivide, OBJ_FEAT, sqrt(sum(OBJ_FEAT .^2, 2)));

%     FEAT = [SC_FEAT, OBJ_FEAT];
    FRM_SC = [SC_FEAT, ones(size(SC_FEAT, 1), 1)] * svmW;
    FRM_SC = FRM_SC';
    

end