function [covariate]= covariate_vector(fv, b_col)
%EEGLAB2BBCI_epo - Convert EEGLAB data into epoched BBCI format
%
%
%Arguments:
%  fv          -     Data structure in BBCI format, containing the field "b_temp"
%  b_col       -     Integer indicating the column of b_temp that ought to be used for the covariate
%  
%
%Returns:
%  covariate   -     1 x 1 x Trial feature array (typically another class index) for which each LDA training will
%                    find an optimal weight, the weight and the feature will be removed before classification.
%                    This is analogous to partializing the covariate out.
%
% 2019-09 AK

b_temp = fv.b_temp;
covariate = b_temp(:,b_col);
covariate = permute(covariate, [3 2 1]); % this makes concatenation with fv.x easier later
covariate = repmat(covariate,1,size(fv.x, 2)); % this makes concatenation with fv.x easier later
end