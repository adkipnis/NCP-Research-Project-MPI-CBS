function [fv_hyp, fv_hyp2]= loadsubjects_cca(sv, optload)
%LoadSubjects_CCA - Load in subject data after CCA analysis, put into BBCI format
%optionally bin, normalize subject by its covariance trace if instructed to concatenate with data from previous subject
%
%Arguments:
%  sv         -      Subject vector (longer than 1 if you want to concatenate subjects, e.g. sv= 1:n_subj)
%  optload    -      Structure with options:
%    'main_path'  -      Path to folder that contains EEG datafiles
%    'file_name'  -      Path that contains the name common to each datafile (sans subject ID)
%    'listing'    -      List with all subject IDs
%    'cond'       -      m x 2 dimensional condition vector with condition indices in B
%                        (trials with indices of each row are put into separate datasets).
%    'norm'       -      Option to normalize data:
%                        0 - no normalization
%                        1 - z-standardization over channels,
%                        2 - division by trace of covariance matrix
%    'srate'      -      Sampling rate
%    'epochwin'   -      Epoch window of loaded dataset
%
%Returns:
%  fv_hyp  -   Large dataset with the trials of all subjects combined
%
% 2019-09 AK

%unpack optload
main_path = optload.main_path; file_name = optload.file_name; listing = optload.listing; norm = optload.norm;
if isfield(optload, 'cond'), cond = optload.cond; end


fv_hyp = []; fv_hyp2 = [];

for s = sv; 
   goal_path= [main_path char(listing(s)) '/CCA_' char(listing(s)) file_name]; %set goal path
   if exist (goal_path) == 0, continue, end
   % 1. load data, re-epoch, and remove specified trials
      EEG = load(goal_path); %load dataset
      
      %create b_temp from scratch
      b_temp = [];
      b_temp(:,2) = EEG.task_perf'; % H, M, FA, CR
      b_temp(:,1) = repmat(s,length(b_temp),1); %subject ID
      b_temp(:,3) = b_temp(:,2) == 1 | b_temp(:,2) == 4; %add accuracy (correct: H, CR)
      b_temp(:,4) = b_temp(:,2) == 1 | b_temp(:,2) == 3; %add intensity judgements (estimated high: H, FA)      
      b_temp(:,5) = repmat(NaN,length(b_temp),1); %unknown reaction times
      b_temp(:,6) = b_temp(:,2) == 1 |  b_temp(:,2) == 2; %add true intensity (true high: H, M)
      EEG.srate = optload.srate;
      EEG.chanlocs(:).labels = [];
      
      %time vector
      win = optload.epochwin; % from s to ms
      EEG.times = win(1):(1/optload.srate):win(2); % time at each sampling point
      EEG.times(end) = []; %remove last point
      %mark trials without response (task_perf == 0) and remove them
      zero_idx = b_temp(:,2) == 0; 
      b_temp(zero_idx,:) = [];
      EEG.CCA_comps(:,:,zero_idx) = []; EEG.data = EEG.CCA_comps;
           
      
   % 2. Convert EEGLAB data to BBCI compatible format 
      % Create labels for true intensity

      fv = eeglab2bbci_epo(EEG, b_temp(:,optload.label_col)', optload.label);
      fv.b_temp = b_temp; % ####### Warning: this step is only for creating a covariate vector later and is not accounted for in splitting the dataset or creating a hypersubject
      
   % 5. Check and correct for unbalanced lables
      %fv = binpercent(fv, 20, 'CP4');    
      
      
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
      fprintf('Finalizing the dataset...\n') 
          fv_hyp = fv;
          if exist ('fv2', 'var'), fv_hyp2 = fv2; end
      else
      fprintf('Concatenating subject data to hypersubject...\n') 
      fv_hyp.x = cat(3,fv_hyp.x, fv.x); %concatenate trials
      fv_hyp.y = cat(2, fv_hyp.y, fv.y); % concatenate labels
      
          if exist ('fv2', 'var')
              fv_hyp2.x = cat(3,fv_hyp2.x, fv2.x); %concatenate trials
              fv_hyp2.y = cat(2, fv_hyp2.y, fv2.y); % concatenate labels
          end
      end 
end

end