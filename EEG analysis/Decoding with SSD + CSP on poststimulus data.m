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


%% Set options in optload, load subjects with SSD components, crossvalidate LDA with CSP 
AUC_cv1 = []; AUC_cv1_tr =[]; AUC_cv1_p = []; AUC_cv1_p_tr =[]; 
AUC_cv2 = []; AUC_cv2_tr =[];AUC_cv2_p = []; AUC_cv2_p_tr =[];
AUC_cv=[]; AUC_cv_tr = [];  AUC_cv_p=[];
Stat = []; Stat_p = []; p = []; p_p = [];

%loading options:
%optload.cond = [1 2; 3 4];% 1 2: only take H and M -> more intense stimulus; 3 4 only take FA and CR -> less intense stimulus
W = 9; %starting frequency
optload.freq = [W W+4; W-2 W+4+2; W-1 W+4+1]; %frequency bands window for SSD
optload.norm = 2; %1: z-standardize continuous data over channels, 2: normalize by dividing each datapoint by the trace of the cov matrix
optload.KeepN = 15; % 0: apply screetest on eigenvalues to keep as the SSD components that account for 90% of the eigenvalue trace
optload.screeplot = 1; %display screeplot of sorted eigenvalues 
optload.filter_order = 2; 
optload.label_col = 7; optload.label = [{'Response'} {'Not Yet'}]; % motor response contained

for s=1:n_subj; %subject vector
[fv, fv2] = loadsubjects_ssd(B, s, optload); 

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
OPTcsp = {};
OPTcsp.SelectFcn = {@procutil_selectMinMax, 6}; %select six largest eigenvalues for each condition
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
cvopt.rep = 3; % 3 repetitions


% Classification with data from first condition
[AUC_cv1_p(s,:), AUC_cv1_p_tr(s,:), p_p(s,1), stat_p] = crossvalidation_AK(fv, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat_p{s,1} = stat_p; clear stat_p;

if size(fv2,1)~= 0, 
% Classification with data from second condition            
[AUC_cv2_p(s,:), AUC_cv2_p_tr(s,:), p_p(s,2), stat_p] = crossvalidation_AK(fv2, cvopt, {@train_RLDAshrink, 'Gamma',0}, ...
                'LossFcn', {@loss_0_1},... 
                'SampleFcn', {@sample_KFold, 10}, ... {@sample_leaveOneOut}
                'Proc', proc); Stat_p{s,2} = stat_p; clear stat_p;
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
observed_data = AUC_cv1(:,1)';
permtest_data = permute(AUC_cv1_p, [3 1 2]);
[Results, Params] = allefeld_algorithm(observed_data, permtest_data);

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

