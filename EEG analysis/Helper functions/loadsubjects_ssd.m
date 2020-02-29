function [fv_hyp, fv_hyp2, A]= loadsubjects_ssd(B, sv, optload)
%loadsubjects - Load in subject data, filter in specified band, put into BBCI format
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
%    'b, a'       -      Bandpass Filter paramaters
%    'cond'       -      m x 2 dimensional condition vector with condition indices in B
%                        (trials with indices of each row are put into separate datasets).
%    'norm'       -      Option to normalize data:
%                        0 - no normalization
%                        1 - z-standardization over channels,
%                        2 - division by trace of covariance matrix
%
%Returns:
%  fv_hyp     -      Potentially large dataset with the trials of all subjects in the sv
%  fv_hyp2    -      Same but for the second condition if 'cond' is used
%  A          -      SSD activation pattern
% 2019-09 AK

%unpack optload
main_path = optload.main_path; file_name = optload.file_name; listing = optload.listing; norm = optload.norm;
if isfield(optload, 'cond'), cond = optload.cond; end
if isfield(optload, 'screeplot') == 0, optload.screeplot = 0; end

fv_hyp = []; fv_hyp2 = []; A =[];

for s = sv; 
   goal_path= [main_path char(listing(s)) '/' char(listing(s)) file_name]; %set goal path
   if exist (goal_path) == 0, continue, end
   % 1. load data, re-epoch, and remove specified trials
      EEG = pop_loadset(goal_path) %load dataset
      b_temp = B(B(:,1)==s, :); %extract subject's behavioral data
      b_temp = b_temp(EEG.etc.accepted_epochs,:); %remove discarded epochs 
      zero_idx = b_temp(:,2) == 0; %mark trials without response (task_perf == 0)
      b_temp(zero_idx,:) = []; EEG.data(:,:,zero_idx) = [];
      %EEG.epoch(:,zero_idx) = [];
      EEG.trials = size(EEG.data, 3);%remove those trials from b_temp and EEG_ep.data  
      
   
   % OLD: 2. Apply SSD on mirrored data
   %fprintf('Mirroring epochs...\n')
   %X = EEG.data;
   %X_s_matrix = zeros(size(X,1), size(X,2)*3, size(X,3));
   
   %parfor k = 1:size(X,3) 
   %     sig = double(X(:,:,k)'); % T*C
   %     sig_mirrored = [flipud(sig);sig;flipud(sig)];
   %     X_s_matrix(:,:,k) = sig_mirrored'; % in order to make parfor work
   %end
   %X = reshape(X_s_matrix, size(X_s_matrix,1), size(X_s_matrix,2)*size(X_s_matrix,3))'; %bring data into SSD format
   %fprintf('Applying separate bandpass filters and calculating SSD...\n')
   %[~, ~, ~, ~, X_ssd] = ssd_mirror(X, optload.freq, EEG.srate, optload.filter_order, [], optload.KeepN);
   %X_ssd = X_ssd'; EEG.data = reshape(X_ssd, size(X_ssd, 1), size(EEG.data, 2), size(EEG.data, 3)); %reshape and store SSD data
   
   % 2. Apply SSD on epoched data
   fprintf('Mirroring epochs, applying separate bandpass filters and calculating SSD...\n')
   [~, A, lambda, ~, X_ssd] = ssd_mirror_concat_par_screetest(EEG.data, optload.freq, EEG.srate, optload.filter_order, [], optload.KeepN);
   EEG.data = X_ssd; %save SSD-filtered data to EEG dataset
   
   if optload.screeplot == 1,
       figure
       bar(lambda) %display screeplot of sorted eigenvalues if instructed
       title(['\bf\fontsize{9} Sorted Eigenvalues of SSD componens for subject #' num2str(s)])
       xlabel('SSD component')
       ylabel('Magnitude of Eigenvalue')       
       ylim([0 1.2])
   end
      
   % 3. Ceteris paribus if instructed
   %if exist('cond', 'var')    
   %  cp_idx = ~ismember(b_temp(:,2), cond); %define trials that are not from the desired condition
   %  b_temp(cp_idx,:) = []; EEG.data(:,:,cp_idx) = []; %remove those trials 
   %end   
      
   % 4. Convert EEGLAB data to BBCI compatible format 
      % Create labels for true intensity
      if optload.label_col == 6
          b_temp(:,6) = b_temp(:,2) == 1 |  b_temp(:,2) == 2; % create col for true intensity (high(1): H & M, low(2): FA & CR)
      end
      
      % Create dataset using the trials from the two extreme tertiles, split each trial, and assign the label of it containing signal from the motor response or not
      if optload.label_col == 7
         [EEG, b_temp]= tertilesplit(EEG, b_temp, optload);
         optload.label_col = 6;
      end
      
      fv = eeglab2bbci_epo(EEG, b_temp(:,optload.label_col)', optload.label); 
      
   % 5. Check and correct for unbalanced lables
      %fv = binpercent(fv, 20, 'CP4');    
      %fv = undersample(fv);
      
   % 6. Normalize
   if length(sv) > 1
   if norm == 1
   % 6.1 z-standardize data along channels
   fprintf('z-standardizing data along channels...\n')
  
      x = permute(fv.x, [2 1 3]);
      z = reshape(x, size(x, 1), size(x, 2)*size(x, 3))';
      z = zscore(z); z = z';
      z = reshape(x, size(x, 1), size(x, 2), size(x, 3));
      fv.x = permute(z, [2 1 3]);
      
   elseif norm == 2   
   % 6.2 normalize data by the trace of its covariance matrix
   fprintf('Normalizing data by the trace of its covariance matrix...\n')
      X = fv.x;
      C_s_matrix = zeros(size(X,2),size(X,2)); % initialize matrix for single-subject covariances
      X_s_tmp = permute(X, [2 1 3]); X_s_tmp = reshape(X_s_tmp, size(X_s_tmp,1), size(X_s_tmp,2) * size(X_s_tmp,3)); % concatenate epochs
      X = X ./ trace(cov(X_s_tmp')); % calculate cov and normalize it by its trace
      fv.x = X; % store normalized data
   end
   end
   
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