function [fv_us]= undersample(fv, vb, to)
%UNDERSAMPLE - Undersampling function for balanced classification
%
%Arguments:
%  fv  -       EEG structure in BBCI format
%  to  -       threshold for tolerated label imbalance
%  vb  -       1: verbos, 0: no messages in command window
%
%Returns:
%  fv_us -     EEG structure in BBCI format with an equal amout of trials for each class
%

% 2019-09 AK

fv_us=fv;

%check if tolerance threshold exists
if exist('to') == 0, to = 50; end 
%check if verbosity is defined
if exist('vb') == 0, vb = 1; end 

%define ratio
ratio = mean(fv.y(1,:), 2); % this will be used later
ratio_percent = round(ratio * 100); % just for informing the user about the ratio
if ratio_percent < 50, ratio_percent = 100 - ratio_percent; end

%decide whether to undersample
if ratio_percent > to,
    if vb == 1, fprintf('~%1$d%% of all samples belong to one class. A balance of %2$d%% was set as tolerable, so the dataset is unbalanced.\nProceeding with undersampling...\n', ratio_percent, to), end
else if vb == 1, fprintf('~%1$d%% of samples belong to one class. A balance of %2$d%% was set as tolerable, so no undersampling will be performed.\n', ratio_percent, to), end
    fv_us.idx_us = logical(ones(size(fv.x, 3),1)); % idx_us with as many components (all ones) as trials in fv.x
    return
end

tn = length(fv.y(1,:)); %trial number
IDX = 1:tn; % all trial indices
IDX_1 = IDX .* fv.y(1,:); IDX_1(IDX_1 == 0) = []; %trial indices of class 1
IDX_0 = setdiff(IDX, IDX_1);  %trial indices of class 0

% define the majority category and the maximal sample size per category 
if ratio > .5 
    max=floor(tn*(1-ratio)); %maximum amount of usable samples per class
    majority = 1; %number of majority class
else
    max=floor(tn*ratio);
    majority = 0;
end

%now randomly sample out of majority set
IDX_us = [];
if majority == 1
   IDX_us = randsample(IDX_1, max); IDX_us = [IDX_us IDX_0];
elseif majority == 0
   IDX_us = randsample(IDX_0, max); IDX_us = [IDX_us IDX_1];
end


% only use sample labels that are indexed in IDX_us
if majority == 1
 fv_us.y = fv.y(1, IDX_us); fv_us.y(2,:) = abs(fv_us.y(1,:)-1);
elseif majority == 0
 fv_us.y = fv.y(2, IDX_us); fv_us.y(2,:) = abs(fv_us.y(1,:)-1);  
end

% make IDX_us logical for later use
idx_us(IDX_us) = logical(1);

fv_us.x = fv.x(:,:,IDX_us); % same for sample features
fv_us.idx_us = idx_us; % save the indices
fv_us.changelog = strcat(fv_us.changelog,'undersampled_ ');
if vb == 1, fprintf('The balanced dataset contains %1$d fewer samples. %2$d samples remain.\n', tn-2*max, 2*max), end

end