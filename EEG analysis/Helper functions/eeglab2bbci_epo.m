function [fv]= eeglab2bbci_epo(EEG, y, cn)
%EEGLAB2BBCI_epo - Convert EEGLAB data into epoched BBCI format
%
%
%Arguments:
%  EEG -    Structure of feature vectors with data in field '.x' and labels in
%           field '.y'. FV.x may have more than two dimensions. The last
%           dimension is assumed to index samples. The labels FV.y must have
%           the format DOUBLE with size nClasses x nSamples.
%
%  y -      T x 1 vector with label (or condition) in 1 or 0
%  cn -     2 cells with Class Names  
%
%Returns:
%  fv -     Data structure in BBCI format
%

% 2019-09 AK

 
fv.fs = EEG.srate; %sampling frequency
fv.clab = {EEG.chanlocs(:).labels}; %channel labels
fv.x = permute(EEG.data,[2 1 3]); %epoched data in S x C x T format
fv.t = EEG.times; % time vector for the epoch window (starting at -windowlength+one sample, ending at 0) in ms in the intervals of the fs
y(2,:) = abs(y-1); fv.y = y; % label vector
fv.className = cn; % class names
fv.changelog = '';
end