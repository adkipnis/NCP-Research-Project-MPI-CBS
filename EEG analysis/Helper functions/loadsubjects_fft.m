function [fv_hyp, fv_hyp2]= loadsubjects_fft(B, sv, optload)
%LoadSubjects_fft - Load in subject data, filter in specified band and extract amplitudes in frequency space, then put into BBCI format
%optionally bin and undersample, normalize subject by its covariance trace,
%optionally concatenate with data from previous subject
%
%Arguments:
%  B          -      Behavior dataset
%  sv         -      Subject vector (longer than 1 if you want to concatenate subjects, e.g. sv= 1:n_subj)
%  optload    -      Structure with options:
%    'main_path'  -      Path to folder that contains EEG datafiles
%    'file_name'  -      Path that contains the name common to each datafile (sans subject ID)
%    'listing'    -      List with all subject IDs
%    'epoc'       -      Specified timewindow for epoching
%    'cond'       -      m x 2 dimensional condition vector with condition indices in B
%                        (trials with indices of each row are put into separate datasets).
%    'norm'       -      Option to normalize data:
%                        0 - no normalization
%                        1 - z-standardization over channels,
%                        2 - division by trace of covariance matrix
%    'mirror'     -      0 - no mirroring before FFT; 1 - mirror before FFT
%    'bfreq'      -      2 x 1 vector with frequency band for which we extract amplitudes to use as features
%
%Returns:
%  fv_hyp  -   Potentially large dataset with the trials of all subjects in the sv
%  fv_hyp2 -   Same but for the second condition if 'cond' is used
% 2019-09 AK

%unpack optload
main_path = optload.main_path; file_name = optload.file_name; listing = optload.listing; norm = optload.norm;
if isfield(optload, 'cond'), cond = optload.cond; end


fv_hyp = []; fv_hyp2 = [];

for s = sv; 
   goal_path= [main_path char(listing(s)) '/' char(listing(s)) file_name]; %set goal path
   if exist (goal_path) == 0, continue, end
   % 1. load data, re-epoch, and remove specified trials
      EEG = pop_loadset(goal_path) %load dataset
      b_temp = B(B(:,1)==s, :); %extract subject's behavioral data
      b_temp = b_temp(EEG.etc.accepted_epochs,:); %remove discarded epochs 
      zero_idx = b_temp(:,2) == 0; %mark trials without response (task_perf == 0)
      b_temp(zero_idx,:) = []; EEG.data(:,:,zero_idx) = [];
      EEG.epoch(:,zero_idx) = []; EEG.trials = size(EEG.data, 3);%remove those trials from b_temp and EEG_ep.data  
      
   
   % 2. OLD: Apply linear filter (redundand if we apply FFT)
   %fprintf('Mirroring and bandpass filtering epochs...\n')

   %X = EEG.data;
   %X_s_matrix = zeros(size(X,1), size(X,2), size(X,3)); 
   %for k = 1:size(X,3) % take only relevant epochs; 
        %k
        %sig = double(X(:,:,k)'); % T*C
        %sig_mirrored = [flipud(sig);sig;flipud(sig)];
        %Pnts = floor(size(sig_mirrored,1)/3); %AH 06/2019
    
        %X_s_epoch = filtfilt(b,a,sig_mirrored); 
        %X_s_epoch = X_s_epoch(Pnts+1:2*Pnts,:);% skip this part for FFT 
       
        %X_s_matrix(:,:,k) = X_s_epoch'; % in order to make parfor work
   %end
   %EEG.data = X_s_matrix; % store filtered data
 
   
      
   % 3. Convert EEGLAB data to BBCI compatible format 
      % Choose labels
      fv = eeglab2bbci_epo(EEG, b_temp(:,4)', [{'High'} {'Low'}]); % y = row vector of intensity ratings 
      %fv = eeglab2bbci_epo(EEG, b_temp(:,3)', [{'Correct'} {'False'}]); % y = row vector of intensity ratings
      
   % 4. Check and correct for unbalanced lables
      %fv = binpercent(fv, 20, 'CP4');    
      %fv = undersample(fv);
      
   % 5. Normalize
   if length(sv) > 1
   if norm == 1
   % 5.1 z-standardize data along channels
   fprintf('z-standardizing data along channels...\n')
  
      x = permute(fv.x, [2 1 3]);
      z = reshape(x, size(x, 1), size(x, 2)*size(x, 3))';
      z = zscore(z); z = z';
      z = reshape(x, size(x, 1), size(x, 2), size(x, 3));
      fv.x = permute(z, [2 1 3]);
      
   elseif norm == 2   
   % 5.2 normalize data by the trace of its covariance matrix
   fprintf('Normalizing data by the trace of its covariance matrix...\n')
      X = fv.x;
      C_s_matrix = zeros(size(X,2),size(X,2)); % initialize matrix for single-subject covariances
      X_s_tmp = permute(X, [2 1 3]); X_s_tmp = reshape(X_s_tmp, size(X_s_tmp,1), size(X_s_tmp,2) * size(X_s_tmp,3)); % concatenate epochs
      X = X ./ trace(cov(X_s_tmp')); % calculate cov and normalize it by its trace
      fv.x = X; % store normalized data
   end
   end
   
   % 6. Extract amplitudes in the frequency domain through FFT
   if optload.mirror == 1
   % 6.1 Mirror signal to increase resolution of FFT
   fprintf('Mirroring epochs before applying FFT...\n')
   X = fv.x; X = permute(X, [2 1 3]);
   X_s_matrix = zeros(size(X,1), size(X,2)*3, size(X,3)); 
     parfor k = 1:size(X,3) % take only relevant epochs;
        %k
        sig = double(X(:,:,k)'); % T*C
        sig_mirrored = [flipud(sig);sig;flipud(sig)];
        X_s_matrix(:,:,k) = sig_mirrored'; % in order to make parfor work
     end
   signal = permute(X_s_matrix, [2 1 3]);
   else 
   signal = fv.x; 
   end
   
   
   
   % 6.2 Apply FFT
   fprintf('Applying FFT and saving amplitudes in frequency domain as new features...\n')   
      signalX = fft(signal) / size(signal, 1); 
      hz = linspace( 0, fv.fs/2, floor(length(signal)/2)+1 ); % frequencies for which the FFT will give us amplitudes; +1 for DC component
      [~, minidx] = min(abs(hz-optload.bfreq(1))); %find index of frequency closest to minimum threshold freq
      [~, maxidx] = min(abs(hz-optload.bfreq(2))); %find index of frequency closest to maximum threshold freq
      hz = hz(minidx:maxidx); %this gives us all frequencies until the frequency in Hz specified in optload.nfreqs
      signalX = signalX(minidx:maxidx,:,:); %only positive frequencies from bfreq (previously only all positive freqs)
      
      if hz(1) == 0,
          amplitudes(1,:,:) = abs(signalX(1,:,:)); %do not double 0 Hz component
          amplitudes(2:length(hz),:,:) = 2*abs(signalX(2:end,:,:)); %double all positive freqs as we discard all negative freqs
      else
          amplitudes(1:length(hz),:,:) = 2*abs(signalX(1:end,:,:)); %double all positive freqs as we discard all negative freqs   
      end
      
      fv.x = amplitudes; %store new features
      fv.hz = hz; %store frequencies
      
      % plot
      % plot(hz,2*abs(signalX(:,1,1)),'p-')
      % title('Frequency domain')
      %ylabel('Amplitude'), xlabel('Frequency (Hz)')

      
   % 7. Split data into two sets (i.e., ceteris paribus if instructed)
   if exist('cond', 'var')    
     fprintf('Splitting data into two sets as specified in ''cond''...\n')   
     fv2 = fv; %copy dataset
     
     %b_temp_us = b_temp(fv.idx_us,:); % take only trials left under undersampling
     b_temp_us = b_temp; %if no undersampling is performed
     idx1 = ismember(b_temp_us(:,2), cond(1,:)); %define trials that are from the first desired condition
     idx2 = ismember(b_temp_us(:,2), cond(2,:)); %define trials that are from the opposite desired condition
     
     fv.x = fv.x(:,:,idx1); fv.y = fv.y(:,idx1); fv.idx_fin = idx1; fv.changelog = strcat(fv.changelog, ['cond_' num2str(cond(1,1)) '_' num2str(cond(1,2)) '_']); 
     fv2.x = fv2.x(:,:,idx2); fv2.y = fv2.y(:,idx2); fv2.idx_fin = idx2; fv2.changelog = strcat(fv2.changelog, ['cond_' num2str(cond(2,1)) '_' num2str(cond(2,2)) '_']);
     
   end      
      
   % 8. Finalize dataset / add to hypersubject
      if size(fv_hyp, 1) == 0 %initialize
          fv_hyp = fv;
          if exist ('fv2', 'var'), fv_hyp2 = fv2; end
      else
      
      fv_hyp.x = cat(3,fv_hyp.x, fv.x); %concatenate trials
      fv_hyp.y = cat(2, fv_hyp.y, fv.y); % concatenate labels
      
          if exist ('fv2', 'var')
              fv_hyp2.x = cat(3,fv_hyp2.x, fv2.x); %concatenate trials
              fv_hyp2.y = cat(2, fv_hyp2.y, fv2.y); % concatenate labels
          end
      end 
      
end

end