%% Preparations
clear all 
cd('/data/pt_01972/Alex/')

addpath('/data/pt_01972/Alex/eeglab2019_0/')
addpath('/data/pt_01972/SPoC_Mina/')
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

%remove subject 13 (no valid data)
listing(13)=[]; n_subj = length(listing);

button_cond = [1 1 2 1 2 1 2 1 2 1 2 1 2 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1]; % sorted by participant number


%% Load in data and extract behavioral markers
B = [];

for s=1:n_subj

   goal_path= [main_path char(listing(s)) '/' char(listing(s)) file_name];
   EEG = pop_loadset(goal_path)
   
   ix_AB = find(ismember({EEG.event.type}, {'A - Out', 'B - Out'}));              
   task_perf = zeros(1, length(ix_AB));
   latency = zeros(1, length(ix_AB));
     
   if button_cond(s) == 1 % strong = left button
        for i = 1 : length(ix_AB) % !!! does not take into account that A-Stimulation might be after A-Out sometimes
            switch EEG.event(ix_AB(i)).type 
                case 'A - Out'
                    latency(i) = EEG.event(ix_AB(i)+1).latency -  EEG.event(ix_AB(i)).latency; %save latency of trial i
                    if strcmp(EEG.event(ix_AB(i)+1).type, '1')
                        task_perf(i) = 1; %hit
                    elseif strcmp(EEG.event(ix_AB(i)+1).type, '2')
                        task_perf(i) = 2; %miss
                    else
                        task_perf(i) = 0; %missing value
                    end
                case 'B - Out'
                    latency(i) = EEG.event(ix_AB(i)+1).latency -  EEG.event(ix_AB(i)).latency; %save latency of trial i
                    if strcmp(EEG.event(ix_AB(i)+1).type, '1')
                        task_perf(i) = 3; %false alarm
                    elseif strcmp(EEG.event(ix_AB(i)+1).type, '2')
                        task_perf(i) = 4; %correct reject
                    else
                        task_perf(i) = 0; %missing value
                    end
            end
        end

   elseif button_cond(s) == 2  
        for i = 1 : length(ix_AB) % !!! does not take into account that A-Stimulation might be after A-Out sometimes
            switch EEG.event(ix_AB(i)).type 
                case 'A - Out'
                    latency(i) = EEG.event(ix_AB(i)+1).latency -  EEG.event(ix_AB(i)).latency; %save latency of trial i
                    if strcmp(EEG.event(ix_AB(i)+1).type, '2')
                        task_perf(i) = 1; %hit
                    elseif strcmp(EEG.event(ix_AB(i)+1).type, '1')
                        task_perf(i) = 2; %miss
                    else
                        task_perf(i) = 0; %missing value
                    end
                case 'B - Out'
                    latency(i) = EEG.event(ix_AB(i)+1).latency -  EEG.event(ix_AB(i)).latency; %save latency of trial i
                    if strcmp(EEG.event(ix_AB(i)+1).type, '2')
                        task_perf(i) = 3; %false alarm
                    elseif strcmp(EEG.event(ix_AB(i)+1).type, '1')
                        task_perf(i) = 4; %correct reject
                    else
                        task_perf(i) = 0; %missing value
                    end
            end
        end
   end

   % sampling points to ms
    latency=latency/5;    
    [EEG_ep, ix_accepted] = pop_epoch(EEG, {'A - Out', 'B - Out'}, [-0.2 -0.005], 'epochinfo', 'yes'); % use second output later for choosing epochs (e.g. in behavioral data) AND prune parameters in DFA!!!!
    zero_idx = task_perf(ix_accepted) == 0; %remove trials without response (task_perf == 0)
    ix_accepted = ix_accepted.*abs(zero_idx-1)'; ix_accepted (ix_accepted == 0) = []; 
    
    
    %subject vector
    subject = ones(1,length(ix_accepted)) * s;
      
    % add accuracies
    accuracies = task_perf == 1 | task_perf == 4;
    
    % add intensity judgement
    judgement = task_perf == 1 | task_perf == 3;
    
    % adjust task performance vector to epochs that were long enough
    behavior = [subject; task_perf(ix_accepted); accuracies(ix_accepted); judgement(ix_accepted); latency(ix_accepted)]';
    
    B = [B; behavior];   
    
end

% safe data set
    save('behavior.mat', 'B')