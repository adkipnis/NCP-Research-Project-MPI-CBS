function [fv_binned]= binpercent(fv, pc, chan)
%BINPERCENT - Sorts trials into bins and creates dataset from both extreme bins only
%
%
%Arguments:
%  fv  -       EEG structure in BBCI format
%  pc  -       Percentile: size of bins
%  chan -      Optional: String with name of the channel from which the data are averaged and sorted for binning
%
%Returns:
%  fv_binned -     EEG structure in BBCI format containing only trials from the desired bins
%  
%
%Example: binpercent(fv, 20, 'CP4')
%
% 2019-09 AK


%check if inputs are correct
if nargin < 2 || nargin > 3 , return; end

% check if input is not binned already
if isfield(fv, 'changelog')
 if contains(fv.changelog, 'binned') 
   fprintf(2,'Warning: This dataset was already binned. Press any key to continue, or Strg+c to quit.\n')
   pause 
 end
end


fv_binned = fv;

%prepare for binning
nbin = 100/pc; %number of bins
ntrials = size(fv.x, 3); % number of samples
nout = mod(ntrials, nbin); % number of "outliers": number of samples that would remain after equal binning
out = randperm(ntrials, nout); %randomly draw the "outliers"
fv.x(:,:,out) = []; fv.y(:,out) = [];%remove these trials
ntrials = size(fv.x, 3); % number of remaining samples
bsize = ntrials/nbin; %bin size


%for binning we need an average value per trial
x = mean(fv.x, 1); %average over samples

if size(chan,1) == 0
  x = squeeze(mean(x, 2))'; %if chan is unspecified, average over all channels
  fprintf('The trials are sorted according to the average value of all channels.\n')

else
  fprintf('The trials are sorted according to the average value of channel %s.\n', chan)
  chan = find(contains(fv.clab, chan)); %use clabs to find index of desired channel
  x = x(:,chan,:); % take data from specified channel
end 


% sort trials in an ascending manner
[x_sort, idx] = sort(x);

%assign trials to bins
for i = 1:nbin
bins(1,i) = {x_sort(1+(bsize*(i-1)) : bsize*i)}; % trial values
bins(2,i) = {idx(1+(bsize*(i-1)) : bsize*i)}; % trial indices
end

%reassemble dataset from both extreme bins
x_left = fv.x(:,:,cell2mat(bins(2,1))); %use indices to sample out of leftmost bin
x_right = fv.x(:,:,cell2mat(bins(2,nbin))); %repeat for rightmost bin
fv_binned.x = cat(3, x_left, x_right); %merge both datasets
y_left = fv.y(:,cell2mat(bins(2,1)));  %repeat for trial labels
y_right = fv.y(:,cell2mat(bins(2,nbin)));
fv_binned.y = cat(2, y_left, y_right); 

fv_binned.changelog = strcat(fv_binned.changelog,'binned_ ');
fprintf('Sorted the samples into %1$d bins and reassembled the dataset using %3$d samples from the 1st and the %2$dth bin each.\n', nbin, nbin, bsize)
end