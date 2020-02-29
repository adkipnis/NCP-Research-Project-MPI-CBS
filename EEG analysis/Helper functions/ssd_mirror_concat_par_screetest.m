function [W, A, lambda, C_s, X_ssd] = ssd_mirror_concat_par_screetest(X, freq, sampling_freq, filter_order, epoch_indices, KeepN)
% Modification 20180813 Tilman and Mina
% X is the epoched data (C*T*E)
%%
%SSD - Spatio-Spectral Decomposition
%
% [W, A, lambda, C_s, X_ssd] = ssd(X, freq, sampling_freq, filter_order, epoch_indices)
%
% This is a function for the extraction of neuronal oscillations 
% with optimized signal-to-noise ratio. The algorithm maximizes 
% the power at the center frequency (signal of interest) while simultaneously suppressing it
% at the flanking frequency bins (considered noise). 
% 
% INPUT: 
%     X -     a matrix of size TxM with T samples and M channels of 
%             raw EEG/MEG/LFP data
%     freq -  3 x 2 matrix with the cut-off frequencies. 
%             First row: cut-off frequencies for band-pass of the to be extracted 
%             oscillations.
%             Second row: cut-off frequencies for the lowest and highest 
%             frequencies defining flanking intervals.
%             Third row: cut-off frequencies for the band-stop filtering of 
%             the central frequency process.
%     sampling_freq -     the sampling frequency (in Hz) of X (the data)
%     filter_order  -     filter order used for butterworth bandpass and 
%                         bandstop filtering. If unsure about it, use [] then
%                         the default value of order = 2 will be used
%     epoch_indices -     a matrix of size N x 2, where N is the number of 
%                         good (i.e. artifact free) data segments. Each row of 
%                         this matrix contains the first and the last sample index of
%                         a data segment, i.e. epoch_indices(n,:) = [1000, 5000]
%                         means that the n'th segment starts at sample 1000 and
%                         ends at sample 5000. If all data is useable or if unsure,
%                         use [], then the default of [1, size(X,1)] will be used.
%     KeepN -             First N SSD filters that will be applied onto the data. If N is not specified,
%                         all filters will be applied. If N is 0, apply a screetest onto the eigenvalues
%                         (take eigenvalues that make out 90% of the eigenvalue trace)
%                         
% OUTPUT:
%     W -     the de-mixing matrix. Each column is a spatial filter and the
%             timecourse of the SSD components is extracted with X * W
%     A -     the spatial patterns (also called mixing matrix) with the i'th column
%             corrresponding to the i'th SSD component
%     lambda - the eigenvalues corresponding to each component. The stronger
%              the eigenvalue the better is the ratio between the signal and noise. 
%              The components are sorted in the descending order (first components 
%              have the largest SNR)
%      C_s -  the covariance matrix of X after bandpass filtering with the band
%             defined in freq(1,:)
%      X_ssd - the bandpass filtered data projected onto the SSD components, 
%              i.e. X_ssd = X_s * W, where X_s is the bandpass filtered version of X
%              
% 
% EXAMPLE:
% Let us consider that we want to extract oscillations in the 10-12 Hz
% frequency range with sampling frequency 200 Hz. 
% Then we define: 
%   freq = [10 12; 8 14; 9 13]. 
% Here we want to extract oscillations in 10-12
% Hz range, and flanking noise is defined as band-pass filtered data in
% 8-14 Hz with the following band-stop filtering in 9-13 Hz in order 
% to prevent spectral leakage to flanking noise from the signal of interest
% (10-12 Hz in this case). 
%   filter_order = 2; 
%   sampling_freq = 200;
% We want only data from 2 seconds to 100 seconds and then from 110 seconds to 150 seconds
% then: 
%   epoch_indices = [2 100; 110 150] .* sampling_freq;
%
% The whole command is then written as:
% [W, A, lambda, C_s, X_ssd] = ssd(X, freq, sampling_freq, filter_order, epoch_indices); 
%
%
% References:
%
% Nikulin VV, Nolte G, Curio G. A novel method for reliable and fast extraction
% of neuronal EEG/MEG oscillations on the basis of spatio-spectral decomposition.
% NeuroImage, 2011, 55: 1528-1535.
%
% Haufe, S., Dahne, S., & Nikulin, V. V. Dimensionality reduction for the 
% analysis of brain oscillations. NeuroImage, 2014 (accepted for publication)
% DOI: 10.1016/j.neuroimage.2014.06.073
%
% 
% 
% Copyright (c) [2014] [Stefan Haufe, Sven Daehne, Vadim Nikulin]
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.


%% check input arguments

% make sure FREQS has the correct dimensions
if not( size(freq,1)==3 && size(freq,2)==2 )
  error('freq must be a 3 by 2 matrix, i.e. three bands must be specified!');
end

% check the given frequency bands
signal_band = freq(1,:); % signal bandpass band
noise_bp_band = freq(2,:); % noise bandpass band
noise_bs_band = freq(3,:); % noise bandstop band
if not( noise_bs_band(1) < signal_band(1) && ...
        noise_bp_band(1) < noise_bs_band(1) && ...
        signal_band(2) < noise_bs_band(2) && ...
        noise_bs_band(2) < noise_bp_band(2) )
  error('Wrongly specified frequency bands!\nThe first band (signal band-pass) must be within the third band (noise band-stop) and the third within the second (noise band-pass)!');
end

% default values for optional arguments
if isempty(filter_order)
    filter_order = 2;
end
if isempty(epoch_indices)
    %epoch_indices = [1, size(X,1)];
    epoch_indices = [1, size(X,3)]; % 3rd dimension are epochs! TS, 06/2019
end

% indices of good segments
ind=[];
for n=1:size(epoch_indices,1)
    ind=[ind epoch_indices(n,1):epoch_indices(n,2)];
end
X = X(:,:,ind); % keep only good segments


%% filtering of data
% Pnts = floor(size(X,2)/3); 
% Creating filters
[b,a]=butter(filter_order, signal_band/(sampling_freq/2));
[b_f,a_f]=butter(filter_order, noise_bp_band/(sampling_freq/2));
[b_s,a_s]=butter(filter_order, noise_bs_band/(sampling_freq/2),'stop');


% Covariance matrix for the center frequencies (signal and noise)
X_s = [];
X_n = [];


X_s_matrix = zeros(size(X,1), size(X,2), length(ind));
X_n_matrix = zeros(size(X,1), size(X,2), length(ind));

parfor k = 1:length(ind) % take only relevant epochs; parfor inserted; TS, 06/2019
    %k
    sig = double(X(:,:,k)'); % T*C
    sig_mirrored = [flipud(sig);sig;flipud(sig)]; % mirroring changed: polarity of mirrored signals flipped for continuity 
    
    Pnts = floor(size(sig_mirrored,1)/3); %AH 06/2019
    
    X_s_epoch = filtfilt(b,a,sig_mirrored); 
    X_s_epoch = X_s_epoch(Pnts+1:2*Pnts,:);% T*C
    %X_s = [X_s;X_s_epoch];    
    
    X_n_epoch = filtfilt(b_f,a_f,sig_mirrored); 
    X_n_epoch = filtfilt(b_s,a_s,X_n_epoch);
    X_n_epoch = X_n_epoch(Pnts+1:2*Pnts,:);% T*C
    %X_n = [X_n;X_n_epoch]; 
    
    X_s_matrix(:,:,k) = X_s_epoch'; % in order to make parfor work
    X_n_matrix(:,:,k) = X_n_epoch'; % in order to make parfor work 
end
% C_s = cov(X_s(ind,:),1);
% C_n = cov(X_n(ind,:),1);

X_s = reshape(X_s_matrix, size(X,1), size(X,2)*length(ind)); % concatenate epochs
X_n = reshape(X_n_matrix, size(X,1), size(X,2)*length(ind)); % concatenate epochs

C_s = cov(X_s');
C_n = cov(X_n');



%% Generalized eigenvalue decomposition

% dim-reduction of X does not have full rank
C = C_s;
[V, D] = eig(C);
[ev_sorted, sort_idx] = sort(diag(D), 'descend');
V = V(:,sort_idx);
% compute an estimate of the rank of the data
tol = ev_sorted(1) * 10^-6;
r = sum(ev_sorted > tol);
if r < size(X,1) % dimension changed; TS & AH, 06/2019
    fprintf('SSD: Input data does not have full rank. Only %d components can be computed.\n',r);
    M = V(:,1:r) * diag(ev_sorted(1:r).^-0.5);
else
    M = eye(size(X,2));
end

if r<KeepN
    KeepN = r;
end
C_s_r = M' * C_s * M;
C_n_r = M' * C_n * M;
[W,D]= eig(C_s_r,C_s_r+C_n_r);
[lambda, sort_idx] = sort(diag(D), 'descend');
W = W(:,sort_idx);

% Alex---------------
if KeepN == 0,
    percent = cumsum(lambda)/sum(lambda); %cumulative percentages of the trace wrt each additional lambda
    [~, KeepN] = min(abs(percent - .9)); %closest to 90%
    fprintf('SSD: %d eigenvalues constitute 90%% of the eigenvalue trace.\n',KeepN);   
end
% ------------------

% Mina---------------
if KeepN > 0
    W = W(:,1:KeepN);
    fprintf('SSD: %d spatial filters will be applied on data.\n',KeepN);  
end

%Mina----------------
W = M * W;
% A is the matrix with the patterns (in columns)
A = C * W / (W'* C * W);


%% apply SSD filters to the data

if nargout > 4
    %X_ssd = X_s * W;
    X_ssd = W' * X_s; % dimensions flipped; TS, 06/2019
    X_ssd = reshape(X_ssd, KeepN, size(X,2), length(ind)); % reshape already here; TS, 06/2019
end



