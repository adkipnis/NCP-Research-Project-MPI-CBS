function [fv_fb]= binfreqs(fv, freq_bins)
%BinFreqs - Bin amplitudes in frequency bins and save them as new features
%
%Arguments:
%  fv         -       EEG structure in BBCI format with fv.x in F x C x T,
%                     and fv.hz containing the frequency indices for each sample in F
%                     (e.g., fv.x(a,b,c) contains the amplitude in the a'th
%                     frequency, fv.hz(1,a), on channel b in trial c)
%  freq_bins  -       m x 2 dimensional vector with m frequency bins in Hz
%                     Note: freq_bins(i, 2) should be = freq_bins(i+1, 1)
%
%Returns:
%  fv_fb      -       EEG structure in BBCI format with fv.x in F x C x T but
%                     F being reduced to m components
%
% 2019-09 AK

fe = []; 
for i=1:length(freq_bins)
  idx{i,:} = find(fv.hz > freq_bins(i,1) & fv.hz <= freq_bins(i,2)); %find indices of values belonging to each bin
  fe(i,:,:) = sum(fv.x(idx{i,:},:,:), 1); %sum values in each bin and save the sum as a new feature
end
fv_fb = fv;
fv_fb.x = fe;
fv_fb.changelog = strcat(fv.changelog, 'freq_binned_');


end
