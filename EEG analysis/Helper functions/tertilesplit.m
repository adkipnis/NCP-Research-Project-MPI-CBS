function [EEG_tert, b_temp_tert]= tertilesplit(EEG, b_temp, optload)
%TertileSplit - Create dataset using the trials from the two extreme tertiles, split each trial,
%               and assign the label of it containing signal from the motor response or not
%
%Arguments:
%  EEG     -       EEG structure in EEGLAB format
%  b_temp  -       Behavioral data array for the corresponding subject
%  optload -       Structure containing loading options from loadsubjects.m
%
%Returns:
%  EEG_tert    -   EEG structure in EEGLAB format after undergoing the TertileSplit procedure
%  b_temp_tert -   Behavioral data array for the corresponding subject after undergoing the TertileSplit procedure
%
% 2019-09 AK

EEG_tert=[]; b_temp_tert = [];

fprintf('Binning data of tertiles of fastest and slowest responses...\n')
first_tertile = prctile(b_temp(:,5), 33); %threshold for earliest 33% of responses
third_tertile = prctile(b_temp(:,5), 67); %%threshold for latest 33% of responses
          
%bin indices
lower_idx = b_temp(:,5)<=first_tertile; %trial indices for first bin
upper_idx = b_temp(:,5)>=third_tertile; %trial indices for second bin
          
%create bins
bin1 = EEG.data(:,:,lower_idx); %trials with early response
bin2 = EEG.data(:,:,upper_idx); %trials with late response
          
%find samples per tertile
pnts_1 = length(EEG.xmin:1/EEG.srate:first_tertile/1000); %amount of samples per bin
pnts_2 = length(third_tertile/1000:1/EEG.srate:EEG.xmax);
          
          
 if pnts_1<pnts_2 == 0 %if the first bin has more points
              pnts_1 = pnts_2; %crop bin one
 else pnts_2 = pnts_1; %otherwise crop bin two
 end
 
pnts = pnts_1-1;
          
%find cutoff sample thresholds
%pnts_total = size(EEG.data, 2);
times = EEG.xmin:1/EEG.srate:EEG.xmax; %timepoints at each sample
[~,first_tertile_thresh] = min(abs(times-first_tertile/1000));
[~,third_tertile_thresh] = min(abs(times-third_tertile/1000));
first_tertile_points = [first_tertile_thresh-pnts first_tertile_thresh]; 
third_tertile_points = [third_tertile_thresh third_tertile_thresh+pnts];
                  
fprintf('Splitting each trial into two equal-sized windows (one containing the response)...\n')
%double each trial by splitting it into two equal-sized cropped trials
bin1_first_tertile = bin1(:,first_tertile_points(1):first_tertile_points(2),:);
bin1_third_tertile = bin1(:,third_tertile_points(1):third_tertile_points(2),:);
bin2_first_tertile = bin2(:,first_tertile_points(1):first_tertile_points(2),:);
bin2_third_tertile = bin2(:,third_tertile_points(1):third_tertile_points(2),:);
          
%behavioral data arrays per bin
b_temp_bin1 = b_temp(lower_idx,:); %crop it too
b_temp_bin2 = b_temp(upper_idx,:);
          
%concatenate b_temp and add index 1 if trials contain a response, 0 else
b_temp = [b_temp_bin1 ones(size(b_temp_bin1,1),1); ...
 b_temp_bin1 zeros(size(b_temp_bin1,1),1); ...
 b_temp_bin2 zeros(size(b_temp_bin2,1),1); ...
 b_temp_bin2 ones(size(b_temp_bin2,1),1)];
 
 
%concatenate EEG data
EEG.data = cat(3, bin1_first_tertile, bin1_third_tertile, bin2_first_tertile, bin2_third_tertile);
EEG.times = times(first_tertile_thresh-pnts:first_tertile_thresh);
EEG.times2 = times(third_tertile_thresh:third_tertile_thresh+pnts);
EEG.pnts = pnts_1; EEG.trials = size(b_temp,1)/2;
EEG_tert = EEG; b_temp_tert = b_temp;
end