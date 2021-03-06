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
%optload.file_name = '_task_pchip_sr5kHz_1to200Hz_vi_averef_nonotch_ICA_removed_epoched_only_responses.set'; % -500 to -10 ms
optload.file_name = '_task_pchip_sr5kHz_1to200Hz_vi_averef_nonotch_ICA_removed_epoched_10to1210ms.set'; % 6000 samples


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


%% Set options in optload, load subjects, crossvalidate LDA with CSP 
AUC_cv1 = []; AUC_cv1_tr =[]; Stat = []; p = []; 
AUC_cv1_p = []; AUC_cv1_p_tr =[]; Stat_p = []; p_p = [];

%design butterworth IIR
optload.filter_order = 2; optload.signal_band = [8 20]; optload.sampling_freq = 5000;
[optload.b,optload.a]=butter(optload.filter_order, optload.signal_band/(optload.sampling_freq/2)); 

%further options
%optload.cond = [1 2; 3 4];% 1 2: only take H and M -> more intense stimulus; 3 4 only take FA and CR -> less intense stimulus
optload.norm = 2; %1: z-standardize continuous data over channels, 2: normalize by dividing each datapoint by the trace of the cov matrix

%optload.label_col = 3; optload.label = [{'Correct'} {'False'}]); %accuracy
%optload.label_col = 4; optload.label = [{'High'} {'Low'}]; %intensity judgement
%optload.label_col = 6; optload.label = [{'True High'} {'True Low'}]; %true intensity 
optload.label_col = 7; optload.label = [{'Response'} {'Not Yet'}]; % motor response contained

for s=1:2; %n_subj; %subject vector
fv = loadsubjects_bp(B, s, optload); 
clear fv_orig
if size(fv,1)==0, continue, end
%save('fv_hyp_int.mat', 'fv', '-v7.3')      

%take the mean prestimulus amplitude
%fvm = fv; fvm.x = mean(fv.x, 1); %not possible if we use variance and logarithm in proc options for crossvalidation 

%% ---- CSP analysis and Classification    
% 1. Common Spatial Pattern analysis
% 1.1 Crossvalidated CSP with LDA        

cvopt.dec = 0; % 0: overall decoding, 3: spatial decoding over time
%cvopt.ds = 100; % downsample signal to 100 Hz (recommended if .dec > 0)
cvopt.us = 1; % 0: no undersampling, 1: undersampling of trials
cvopt.rep = 1; % 1: 1 repetition
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

% downsample if instructed
if isfield (cvopt, 'ds') == 1 & ~exist ('fv_orig', 'var')
    fv_orig = fv;
    fv = pop_resample_bbci(fv, cvopt.ds); %this is a slightly rewritten version of EEGLAB's pop_resample
    fprintf('Now each trial has %d datapoints per channel.\n', size(fv.x, 1));
end

[AUC_cv1(s,:), AUC_cv1_tr(s,:), p(s,1), stat, A_tmp] = crossvalidation_AK(fv, cvopt, {@fitcsvm, 'KernelFunction', 'rbf',...
                'KernelScale','auto', 'Standardize',true, 'OptimizeHyperparameters', 'none', 'Verbose', 0}, ...
                'LossFcn', {@loss_0_1 @loss_sensitivity @loss_specificity @loss_rocArea},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat{s,1} = stat; clear stat; A{s,1} = A_tmp;


% Classification with data from first condition
if cvopt.dec < 3
[AUC_cv1(s,:,:), AUC_cv1_tr(s,:,:), p(s,1), stat, A_tmp] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1 @loss_sensitivity @loss_specificity @loss_rocArea},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat{s,1} = stat; clear stat; A{s,1} = A_tmp;
else
[AUC_cv1{s,1}, AUC_cv1_tr{s,1}, p{s,1}, stat] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1 @loss_sensitivity @loss_specificity @loss_rocArea},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat{s,1} = stat; clear stat; 
end

% -------------------------------------------------------------------------


% 1.2 Crossvalidated CSP with LDA with permuted labels       
cvopt.perm = 1; % 0: standard procedure, 1: permute labels
cvopt.rep = 3; % 3 repetitions


% Classification with data from first condition
if cvopt.dec < 3
[AUC_cv1_p(s,:,:), AUC_cv1_p_tr(s,:,:), p_p(s,1), stat_p] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat_p{s,1} = stat_p; clear stat_p;
else
[AUC_cv1_p{s,1}, AUC_cv1_p_tr{s,1}, p_p{s,1}, stat_p] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat_p{s,1} = stat_p; clear stat_p;
end
            
end

% Subject statistics
AUC_cv1(13,:) = []; AUC_cv1_tr(13,:) = []; AUC_cv1_p(13,:) = [];
AUC_cv = AUC_cv1;
AUC_cv_tr = AUC_cv1_tr;
AUC_cv_p = AUC_cv1_p;
mean(AUC_cv,1)
mean(AUC_cv_tr,1)
sig=p(:,1)<0.01;
find(sig)

% 2.1 Group-level statistical inference (Allefeld et al., 2016)
observed_data = AUC_cv(:,1)';
permtest_data = permute(AUC_cv_p', [3 2 1]);
[Results, Params] = allefeld_algorithm(observed_data, permtest_data);

% Plot for spatial decoding over time
if cvopt.dec == 3  
curve = NaN(60,2); % to circumvent dimension mismatch
 for s=1:2 %n_subj
     curve_tmp = squeeze(AUC_cv1{s,1}(:,1,:));
     curve(1:length(curve_tmp),s)  = curve_tmp;
 end
 plot(nanmean(curve, 2))
end

% Plot subject statistics
[sortedacc, sortidx] = sortrows(AUC_cv(:,1:3),1); %sortidx = categorical(string(sortidx));
sortedperm = mean(AUC_cv_p,2); sortedperm = sortedperm(sortidx,:);
sortedall = [sortedacc sortedperm];

figure
h= bar(sortedall, 'BaseValue',50);
title({'{\bf\fontsize{14} Classification of motor reaction presence}'; '\itTo test the viability of the decoding pipeline'});
xlabel('Participant ID, sorted according to Accuracy');
ylabel('Classification performance');
legend('Accuracy','Sensitivity', 'Specificity', 'Accuracy with permuted labels');

%% CSP Activation patterns Plotting   
           
      %Prepare for plotting
      mnt = mnt_setElectrodePositions(fv.clab);
      mnt.pos_3d(1,59) = 0; mnt.x(59) = 0; % fpz has wrong X coordinate (should be 0)
      mnt.pos_3d(3,59) = mnt.pos_3d(2,1); mnt.y(59) = mnt.y(1); % take y-coordinate fp1
      mnt.pos_3d(2,59) = mnt.pos_3d(2,1); % take height from fp1
           
      %clf
      %text(mnt.x, mnt.y, mnt.clab); 
      %axis([-1 1 -1 1]) %just to go sure
      n = size(A,2); plopt.CLim = [-.6 .6];
      
      % plot SSD components for one subject
      n_plots = 2; %subject
      for s = 1:n_plots
          figure
          
          for N = 1:n 
             subplot(2,3,N)
             plot_scalp(mnt, A{s,1}(:,N), plopt);
             title(['\bf\fontsize{10} Mean activation pattern #' num2str(N)]);
             hold on
          end 
          
          hold off
      end
          
