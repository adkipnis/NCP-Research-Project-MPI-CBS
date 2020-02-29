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


%% Set options in optload, load subjects with FFT features, crossvalidate LDA with CSP 
AUC_cv1 = []; AUC_cv1_tr =[]; AUC_cv2 = []; AUC_cv2_tr =[]; Stat = []; p = [];

%freq_bins = [0.5 4; 4 8; 8 14; 14 30; 30 50; 50 100]; %delta, theta, alpha, beta, lower gamma, upper gamma

%further options
optload.cond = [1 2; 3 4];% 1 2: only take H and M -> more intense stimulus; 3 4 only take FA and CR -> less intense stimulus
optload.norm = 2; %1: z-standardize continuous data over channels, 2: normalize by dividing each datapoint by the trace of the cov matrix
optload.bfreq = [4 30]; % Take freqs between first and second variable in Hz
optload.mirror = 0; % 0: don't mirror, 1: mirror before FFT

for s=1:n_subj; %subject vector
[fv, fv2] = loadsubjects_fft(B, s, optload);
if size(fv,1)==0, continue, end
%save('fv_hyp_int.mat', 'fv', '-v7.3')    

% bin amplitudes in freq bins and save them as new features
%fv = binfreqs(fv, freq_bins);
%fv2 = binfreqs(fv2, freq_bins);

cvopt.dec = 0; % 0: overall decoding (feature vector of samples x channels) 
cvopt.rep = 3; % 1: 1 repetition
cvopt.us = 1; % 0: no undersampling, 1: undersampling
cvopt.perm = 0; % 0: standard procedure, 1: permute labels

% Classification with data from first condition
[AUC_cv1(s,:), AUC_cv1_tr(s,:), p(s,1), stat] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},...  %{@loss_rocArea}
                'SampleFcn', {@sample_KFold, 10}); %{@sample_leaveOneOut}
                 Stat{s,1} = stat; clear stat;

% Classification with data from second condition            
[AUC_cv2(s,:), AUC_cv2_tr(s,:), p(s,2), stat] = crossvalidation_AK(fv2, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},...  %{@loss_rocArea}
                'SampleFcn', {@sample_KFold, 10}); % {@sample_leaveOneOut}
                 Stat{s,2} = stat; clear stat;      

% -------------------------------------------------------------------------


% 1.2 Crossvalidated CSP with LDA with permuted labels       
cvopt.perm = 1; % 0: standard procedure, 1: permute labels


% Classification with data from first condition
[AUC_cv1_p(s,:), AUC_cv1_p_tr(s,:), p_p(s,1), stat_p] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},...  %{@loss_rocArea}
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat_p{s,1} = stat_p; clear stat_p;

% Classification with data from second condition            
[AUC_cv2_p(s,:), AUC_cv2_p_tr(s,:), p_p(s,2), stat_p] = crossvalidation_AK(fv2, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},...  %{@loss_rocArea}
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat_p{s,2} = stat_p; clear stat_p;
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

% 2.1 Group-level statistical inference (Allefeld et al., 2016)
observed_data = AUC_cv';
permtest_data = permute(AUC_cv_p, [3 1 2]);
[Results, Params] = allefeld_algorithm(observed_data, permtest_data);
