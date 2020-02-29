%% Preparations
clear all 
cd('/data/pt_01972/Alex/CSP/')
addpath('/data/pt_01972/Alex/eeglab2019_0/')
addpath(genpath('/data/pt_01972/Alex/CSP/'))
startup_bbci_toolbox('DataDir','/data/pt_01972/Alex/CSP/bbci_public-master/data');
eeglab

optload = [];
%create pathlist for participants
optload.main_path = '/data/pt_01972/Preproc_data/N20_study1/';
%file_name = '_task_pchip_sr5kHz_1to200Hz_vi_averef_nonotch_ICA_removed.set';
optload.file_name = '_task_pchip_sr5kHz_1to200Hz_vi_averef_nonotch_ICA_removed_epoched_only_responses.set';
n_subj = 33;
optload.listing = dir(optload.main_path); 
optload.listing=struct2cell(optload.listing)';
optload.listing(:,2:end)=[];
optload.listing(1:2,:)=[];
optload.listing(n_subj+1:end,:)=[]; 
%remove subject 13 (no valid data)
optload.listing(13)={''}; 

%load in behavioral data
load('behavior_CSP.mat')

%% Set options in optload, load subjects with SSD components, crossvalidate LDA with CSP 
AUC_cv1 = []; AUC_cv1_tr =[]; AUC_cv1_p = []; AUC_cv1_p_tr =[]; 
AUC_cv2 = []; AUC_cv2_tr =[];AUC_cv2_p = []; AUC_cv2_p_tr =[];
AUC_cv=[]; AUC_cv_tr = [];  AUC_cv_p=[];
Stat = []; Stat_p = []; p = []; p_p = [];

%loading options:
%optload.cond = [1 2; 3 4];% 1 2: only take H and M -> more intense stimulus; 3 4 only take FA and CR -> less intense stimulus
optload.norm = 2; %1: z-standardize continuous data over channels, 2: normalize by dividing each datapoint by the trace of the cov matrix
optload.freq = [8 13; 6 15; 7 14]; %for SSD
optload.KeepN = 0; % apply screetest on eigenvalues to keep as the SSD components that account for 90% of the eigenvalue trace
optload.screeplot = 1; %display screeplot of sorted eigenvalues 
optload.filter_order = 2; 


for s=1:n_subj; %subject vector
    
[fv, fv2, a] = loadsubjects_ssd(B, s, optload); %load in subject s and apply SSD
if size(fv,1)==0, continue, end
%save('fv_hyp_int.mat', 'fv', '-v7.3') %if you want to save the hypersubject dataset    
A{s,:} = a; clear a;

%% CSP analysis and Classification    
% 1.1 Crossvalidated CSP with LDA        

cvopt.dec = 0; % 0: overall decoding
cvopt.rep = 3; % 1: 1 repetition
cvopt.us = 1; % 0: no undersampling, 1: undersampling
cvopt.perm = 0; % 0: standard procedure, 1: permute labels
OPTcsp = {};
OPTcsp.SelectFcn = {@procutil_selectMinMax, 3}; %select three largest eigenvalues for each condition
proc.train= {{'CSP', @proc_csp, OPTcsp} %type in CSP function and its option structure
                @proc_variance
                @proc_logarithm
                };
proc.apply= {{@proc_linearDerivation, '$CSP'}
                @proc_variance
                @proc_logarithm
                };

% Classification with data from first condition
[AUC_cv1(s,:), AUC_cv1_tr(s,:), p(s,1), stat] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1 @loss_sensitivity @loss_specificity @loss_rocArea},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat{s,1} = stat; clear stat;

if size(fv2,1)~= 0, 
% Classification with data from second condition            
[AUC_cv2(s,:), AUC_cv2_tr(s,:), p(s,2), stat] = crossvalidation_AK(fv2, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1 @loss_sensitivity @loss_specificity @loss_rocArea},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat{s,2} = stat; clear stat;      
end

% -------------------------------------------------------------------------


% 1.2 Crossvalidated CSP with LDA with permuted labels       
cvopt.perm = 1; % 0: standard procedure, 1: permute labels


% Classification with data from first condition
[AUC_cv1_p(s,:), AUC_cv1_p_tr(s,:), p_p(s,1), stat_p] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},...  %{@loss_rocArea}
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat_p{s,1} = stat_p; clear stat_p;

if size(fv2,1)~= 0, 
% Classification with data from second condition            
[AUC_cv2_p(s,:), AUC_cv2_p_tr(s,:), p_p(s,2), stat_p] = crossvalidation_AK(fv2, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},...  %{@loss_rocArea}
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat_p{s,2} = stat_p; clear stat_p;
end
end

% Subject statistics
AUC_cv1(AUC_cv1==0) = []; AUC_cv1_tr(AUC_cv1_tr==0) = []; %p(~any(p,2), :) = [];
AUC_cv2(AUC_cv2==0) = []; AUC_cv2_tr(AUC_cv2_tr==0) = []; p(13,:)=[];
AUC_cv = (AUC_cv1+AUC_cv2)/2;
AUC_cv_tr = (AUC_cv1_tr+AUC_cv2_tr)/2;
mean(AUC_cv)
mean(AUC_cv_tr)
sig=p(:,1)<0.05 & p(:,2)<0.05
find(sig)
AUC_cv_p = (AUC_cv1_p+AUC_cv2_p)/2; AUC_cv_p(13,:) = [];

% Plot subject statistics
[sortedacc, sortidx] = sort(AUC_cv); %sortidx = categorical(string(sortidx));
sortedperm = mean(AUC_cv_p,2); sortedperm = sortedperm(sortidx,:);
sortedacc = [sortedacc sortedperm];
bar(sortedacc, 'BaseValue',50)
bar(mean(sortedacc), 'BaseValue',50)

% 2.1 Group-level statistical inference (Allefeld et al., 2016)
observed_data = AUC_cv';
permtest_data = permute(AUC_cv_p, [3 1 2]);
[Results, Params] = allefeld_algorithm(observed_data, permtest_data);

% 2.2 CSP eigenvalue permutation test
OPTp.np = 100; %100 permutations 
CSP = perm_CSP_eigen(fv, OPTcsp, OPTp); 
disp(CSP.Pval);

           
%% SSD Plotting   
           
      %Prepare for plotting
      mnt = mnt_setElectrodePositions(fv.clab);
      mnt.pos_3d(1,59) = 0; mnt.x(59) = 0; % fpz has wrong X coordinate (should be 0)
      mnt.pos_3d(3,59) = mnt.pos_3d(2,1); mnt.y(59) = mnt.y(1); % take y-coordinate fp1
      mnt.pos_3d(2,59) = mnt.pos_3d(2,1); % take height from fp1
           
      %clf
      %text(mnt.x, mnt.y, mnt.clab); 
      %axis([-1 1 -1 1]) %just to go sure
      n = size(fv.x, 2); plopt.CLim = [-.5 .5];
      
      % plot SSD components for one subject
      s = 1; %subject
      for N = 1:n 
          subplot(ceil(n/6),6,N)
          plot_scalp(mnt, A{s}(:,N), plopt);
          hold on
      end 
      hold off
          

      
  