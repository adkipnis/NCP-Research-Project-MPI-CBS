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
optload.file_name = '_task_pchip_sr5kHz_1to200Hz_vi_averef_nonotch_ICA_removed_epoched_-200to-10ms.set';


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
AUC_cv1 = []; AUC_cv1_tr =[]; AUC_cv2 = []; AUC_cv2_tr =[]; Stat = []; p = [];

%design butterworth FIR
optload.filter_order = 2; optload.signal_band = [8 13]; optload.sampling_freq = 5000;
[optload.b,optload.a]=butter(optload.filter_order, optload.signal_band/(optload.sampling_freq/2)); 

%further options
%optload.cond = [1 2; 3 4];% 1 2: only take H and M -> more intense stimulus; 3 4 only take FA and CR -> less intense stimulus
optload.norm = 2; %1: z-standardize continuous data over channels, 2: normalize by dividing each datapoint by the trace of the cov matrix
%optload.label_col = 3; optload.label = [{'Correct'} {'False'}]); %accuracy
optload.label_col = 4; optload.label = [{'High'} {'Low'}]; %intensity judgement
%optload.label_col = 6; optload.label = [{'True High'} {'True Low'}]; %true intensity 

for s=1:n_subj; %subject vector
[fv, fv2] = loadsubjects_bp(B, s, optload); 

if size(fv,1)==0, continue, end
%save('fv_hyp_int.mat', 'fv', '-v7.3')      

%take the mean prestimulus amplitude
%fvm = fv; fvm.x = mean(fv.x, 1); %not possible if we use variance and logarithm in proc options for crossvalidation 

%% ---- CSP analysis and Classification    
% 1. Common Spatial Pattern analysis
% 1.1 Crossvalidated CSP with LDA        

cvopt.dec = 0; % 0: overall decoding
cvopt.rep = 1; % 1: 1 repetition
cvopt.us = 1; % 0: no undersampling, 1: undersampling
cvopt.perm = 0; % 0: standard procedure, 1: permute labels
cvopt.covariate = covariate_vector(fv, 6); %extract the column number 3 (accuracy) from b_temp in fv
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
[AUC_cv1(s,:), AUC_cv1_tr(s,:), p(s,1), stat, A_tmp] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1 @loss_sensitivity @loss_specificity @loss_rocArea},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat{s,1} = stat; clear stat; A{s,1} = A_tmp;

% Classification with data from second condition            
%[AUC_cv2(s,:), AUC_cv2_tr(s,:), p(s,2), stat] = crossvalidation_AK(fv2, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
%                'LossFcn', {@loss_0_1 @loss_sensitivity @loss_specificity @loss_rocArea},... 
%                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
%                'Proc', proc); Stat{s,2} = stat; clear stat;        
            
% -------------------------------------------------------------------------


% 1.2 Crossvalidated CSP with LDA with permuted labels       
cvopt.perm = 1; % 0: standard procedure, 1: permute labels


% Classification with data from first condition
[AUC_cv1_p(s,:), AUC_cv1_p_tr(s,:), p_p(s,1), stat_p] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},...  %{@loss_rocArea}
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat_p{s,1} = stat_p; clear stat_p;

% Classification with data from second condition            
%[AUC_cv2_p(s,:), AUC_cv2_p_tr(s,:), p_p(s,2), stat_p] = crossvalidation_AK(fv2, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
%                'LossFcn', {@loss_0_1},...  %{@loss_rocArea}
%                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
%                'Proc', proc); Stat_p{s,2} = stat_p; clear stat_p;
fprintf('\n\n')
            
end

% Subject statistics
AUC_cv1(13,:) = []; AUC_cv1_tr(13,:) = []; %AUC_cv_p(13,:) = [];
mean(AUC_cv1)
mean(AUC_cv1_tr)
sig=p(:,1)<0.05 %& p(:,2)<0.05
find(sig)


% 2.1 Group-level statistical inference (Allefeld et al., 2016)
%observed_data = AUC_cv';
%permtest_data = permute(AUC_cv_p, [3 1 2]);
%[Results, Params] = allefeld_algorithm(observed_data, permtest_data);


% Plot subject statistics
[sortedacc, sortidx] = sortrows(AUC_cv1(:,1:3),1); %sortidx = categorical(string(sortidx));
%sortedperm = mean(AUC_cv_p,2); sortedperm = sortedperm(sortidx,:);
%sortedall = [sortedacc sortedperm];
figure
h= bar(sortedacc, 'BaseValue',50);
title({'{\bf\fontsize{14} Classification of intensity rating}'; '\itWith "true intensity" as a covariate'});
xlabel('Participant ID, sorted according to Accuracy');
ylabel('Classification performance');
legend('Accuracy','Sensitivity', 'Specificity');


           
%% Plotting   
           
      %Prepare for plotting
      mnt = mnt_setElectrodePositions(fv_csp.origClab);
      mnt.pos_3d(1,59) = 0; mnt.x(59) = 0; % fpz has wrong X coordinate (should be 0)
      mnt.pos_3d(3,59) = mnt.pos_3d(2,1); mnt.y(59) = mnt.y(1); % take y-coordinate fp1
      mnt.pos_3d(2,59) = mnt.pos_3d(2,1); % take height from fp1
           
      %clf
      %text(mnt.x, mnt.y, mnt.clab); 
      %axis([-1 1 -1 1]) %just to go sure
      n = size(fv_csp.x, 2); plopt.CLim = [-.5 .5];
      
      % Actual plotting
      for N = 1:n 
          subplot(2,3,N)
          plot_scalp(mnt, CSP_A(:,N), plopt);
          hold on
      end 
      hold off
          
%% Historical
%% ---- BCCI functions: Classification and CSP analysis 
% 0.1 Classification without CSP 
%[AUC_raw, AUC_raw_tr]= crossvalidation(fv, @train_RLDAshrink, 'sampleFcn', {@sample_KFold, 10}, 'LossFcn',@loss_rocArea);
%[AUC_raw, AUC_raw_tr]= crossvalidation(fvm, @train_RLDAshrink, 'sampleFcn', {@sample_KFold, 10}, 'LossFcn',@loss_rocArea);
      
% 0.2 First CSP, then Classification (wrong approach)
%OPTcsp = {};
%OPTcsp.SelectFcn = {@procutil_selectMinMax, 1}; %select most negative and positive eigenvalues
%[fv_csp, CSP_W, CSP_A]= proc_csp(fv, OPTcsp);
%[AUC_nocv, AUC_nocv_tr]= crossvalidation_AK(fv_csp, 0, @train_RLDAshrink, 'sampleFcn', {@sample_KFold, 10}, 'LossFcn',@loss_rocArea);
      
%% ---- rudimentary k-folds
%OPTkf.k = 10; OPTkf.rep = 1; OPTkf.dec = 0; %(0) Overall, (1) Spatial, (2) Temporal, (3) Spatio-temporal  
%[AUC_cv2, AUC_cv2_tr] = kfcv_CSP(fv, OPTcsp, OPTkf); % 10-fold CV function with 5 repetitions, train LDA on CSP trained data, test it on test data within each fold
%disp(AUC_cv2); disp(AUC_cv2_tr); if OPTkf.dec == 3, plot(AUC), end;

% calculate average amplitude within prestimulus period and check if 
%aa = mean(DAT.x, 1); %%average alpha
% p = squeeze(aa(:,1,:));
      
  