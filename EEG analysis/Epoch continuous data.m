%% Preparations
clear all 
cd('/data/pt_01972/Alex/CSP/')
addpath('/data/pt_01972/Alex/eeglab2019_0/')
addpath('/data/pt_01972/Alex/')
eeglab

%create pathlist for participants
main_path = '/data/pt_01972/Preproc_data/N20_study1/';
file_name = '_task_pchip_sr5kHz_1to200Hz_vi_averef_nonotch_ICA_removed.set';
n_subj = 33;
listing = dir(main_path); 
listing=struct2cell(listing)';
listing(:,2:end)=[];
listing(1:2,:)=[];
listing(n_subj+1:end,:)=[]; 
listing(13)={''};%remove subject 13 (no valid data)

epoc = [.01 1.21]; %epoch size
%new_fn= '_task_pchip_sr5kHz_1to200Hz_vi_averef_nonotch_ICA_removed_epoched_-200to-10ms.set';
%new_fn= '_task_pchip_sr5kHz_1to200Hz_vi_averef_nonotch_ICA_removed_epoched_200to700ms.set';
new_fn= '_task_pchip_sr5kHz_1to200Hz_vi_averef_nonotch_ICA_removed_epoched_10to1210ms.set'; % 6000 samples
   
%% Load in continuous data and save epoched data

for s = 1:n_subj

   goal_path= [main_path char(listing(s)) '/' char(listing(s)) file_name];
   if exist (goal_path) == 0, continue, end
   EEG = pop_loadset(goal_path)              
   
   %Used create epoched EEG datasets 
   [EEG, ix_accepted] = pop_epoch(EEG, {'A - Out', 'B - Out'}, epoc, 'epochinfo', 'yes'); % use second output later for choosing epochs (e.g. in behavioral data) AND prune parameters in DFA!!!!
    EEG.etc.accepted_epochs = ix_accepted;
    pop_saveset(EEG, 'filename', [main_path char(listing(s)) '/' char(listing(s)) new_fn]);
       
end
   