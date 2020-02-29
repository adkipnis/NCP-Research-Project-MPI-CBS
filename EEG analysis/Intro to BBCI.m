%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Gentle Introduction to the BBCI Toolbox for Offline Analysis %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
addpath('/data/pt_01972/Alex/bbci_public-master/')
startup_bbci_toolbox('DataDir','/data/pt_01972/Alex/bbci_public-master/data');

%% Exploring the data structure cnt

file= fullfile(BTB.DataDir, 'demoRaw', 'VPiac_10_10_13/calibration_CenterSpellerMVEP_VPiac');
[cnt, vmrk]= file_readBV(file, 'Fs',100);
% -> information in help shows how to define a filter

% data structure of continuous signals
cnt
% Fields of cnt - most important are clab (channel labels), x (data) and fs (sampling rate).    
% The data cnt.x is a two dimensional array with dimension TIME x CHANNELS.
% This is a cell array of strings that hold the channel labels. It corresponds to the second dimension of cnt.x.
cnt.clab
% Index of channel Cz
strmatch('Cz', cnt.clab)
% Index of channel P1? Why are there two?
strmatch('P1', cnt.clab)
cnt.clab([49 55])
% Ah, it matches also P10. To avoid that, use option 'exact' in strmatch
strmatch('P1', cnt.clab, 'exact')
% Now it works. The toolbox function 'chanind' does exact match by default:
util_chanind(cnt, 'P1')
% But it is also more powerful. And hopefully intuitive:
idx= util_chanind(cnt, 'P*')
cnt.clab(idx)
idx= util_chanind(cnt, 'P#')
cnt.clab(idx)
idx= util_chanind(cnt, 'P5-6')
cnt.clab(idx)
idx= util_chanind(cnt, 'P3,z,4')
cnt.clab(idx)
idx= util_chanind(cnt, 'F3,z,4', 'C#', 'P3-4')
cnt.clab(idx)
% Be aware that intervals like 'P3-4' correspond to the rows on the scalps layout, i.e.
% P3-4 -> P3, P1, Pz, P2, P4; and F7-z -> F7,F5,F3,F1,Fz
% Furthermore, # matches all channels in the respective row: but C# does not match CP2.
% (but not the temporal locations, i.e. CP# does not match TP7)

%% The montage structure mnt defining electrode layout 

% data structure defining the electrode layout
% Fields of mnt, most importantly clab, x, y which define the electrode montage.
mnt= mnt_setElectrodePositions(cnt.clab)
mnt.clab
% x-coordinates of the first 10 channels in the two projects (from above with nose pointing up).
mnt.x(1:10)
% and the corresponding y coordinates.
mnt.y(1:10)
clf
text(mnt.x, mnt.y, mnt.clab); 
% displays the electrode layout. 'axis equal' may required to show it in the right proportion.
% The function scalpPlot can be used to display distributions (e.g. of voltage) on the scalp:
axis([-1 1 -1 1])
plot_scalp(mnt, cnt.x(200,:));
% The function plot_scalp is kind of a low level function. There is no
% mechanisms that can guarrantee the correct association of the channels
% from the map with the channels in the electrode montage. The user has
% to take care of this (or use another function).
% The following lines demonstrate the issue:
plot_scalp(mnt, cnt.x(200,1:63))
% -> throws an error
% If you use a subset of channels, specify channel labels in 'WClab':
plot_scalp(mnt, cnt.x(200,1:63), 'WClab',cnt.clab(1:63))
plot_scalp(mnt, cnt.x(200,[1:30 35:64]), 'WClab',cnt.clab([1:30 35:64]))
% This results in a wrong mapping:
plot_scalp(mnt, cnt.x(200,[1:30 35:64]), 'WClab',cnt.clab([1:60]))

%% The Marker structure mrk

% data structure defining the markers (trigger events in the signals)
vmrk
vmrk.event.desc(1:100)
classDef= {31:46, 11:26; 'target', 'nontarget'};
mrk= mrk_defineClasses(vmrk, classDef);
mrk.y(:,1:40)
% row 1 of mrk.y defines membership to class 1, and row 2 to class 2
mrk.event.desc(1:40)
sum(mrk.y,2)
it= find(mrk.y(1,:));
it(1:10)
% are the indices of target events

%% Segmentation and plotting of ERPs

% segmentation of continuous data in 'epochs' based on markers
epo= proc_segmentation(cnt, mrk, [-200 800])
epo
iCz= util_chanind(epo, 'Cz') % find the index of channel Cz
plot(epo.t, epo.x(:,iCz,1))
xlabel('time  [ms]');
ylabel('potential @Cz  [\muV]');
plot(epo.t, epo.x(:,iCz,2))
plot(epo.t, epo.x(:,iCz,it(1))) %Plot the EEG trace of an evoked potential
plot(epo.t, epo.x(:,iCz,it(2)))
% The ERP of the target trials is obtained by averaging across all target trials:
plot(epo.t, mean(epo.x(:,iCz,it),3)) %Plot the ERP of channel Cz
in= find(mrk.y(2,:));
hold on
plot(epo.t, mean(epo.x(:,iCz,in),3), 'k')
% you should put labels to the axes, but ...

% visualization of ERPs with toolbox functions
plot_channel(epo, 'Cz')
grid_plot(epo);
grd= sprintf('F3,Fz,F4\nC3,Cz,C4\nP3,Pz,P4')
mnt= mnt_setGrid(mnt, grd);
grid_plot(epo, mnt, defopt_erps);
grd= sprintf(['scale,_,F5,F3,Fz,F4,F6,_,legend\n' ...
              'FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8\n' ...
              'T7,C5,C3,C1,Cz,C2,C4,C6,T8\n' ...
              'P7,P5,P3,P1,Pz,P2,P4,P6,P8\n' ...
              'PO9,PO7,PO3,O1,Oz,O2,PO4,PO8,PO10']);
mnt= mnt_setGrid(mnt, grd);
% One can also use template grids with
mnt= mnt_setGrid(mnt, 'M');
%
grid_plot(epo, mnt, defopt_erps);
% baseline drifts:
clf; plot(cnt.x(1:1000,32))
% To get rid of those, do a baseline correction
epo= proc_baseline(epo, [-200 0]);
H= grid_plot(epo, mnt, defopt_erps);
epo_auc= proc_aucValues(epo);
grid_addBars(epo_auc, 'HScale',H.scale);

%% Plotting scalp topographies epo_auc= proc_aucValues(epo);
ival= select_time_intervals(epo_auc, 'visualize', 1, 'visu_scalps', 1, ...
                            'nIvals',5);
fv= proc_jumpingMeans(epo, ival);
loss_spatioTemp = crossvalidation(fv, @train_RLDAshrink, 'sampleFcn', {@sample_KFold, 5}, 'LossFcn',@loss_rocArea);


% visualization of scalp topograhies
plot_scalpEvolutionPlusChannel(epo, mnt, {'Cz','PO7'}, [200:50:500], defopt_scalp_erp);
figure(2);
plot_scalpEvolutionPlusChannel(epo_auc, mnt, {'Cz','PO7'}, [200:50:500], defopt_scalp_r);
% refine intervals
ival= [250 300; 350 400; 420 450; 490 530; 700 740];
plot_scalpEvolutionPlusChannel(epo_auc, mnt, {'Cz','PO7'}, ival, defopt_scalp_r);
figure(1);
plot_scalpEvolutionPlusChannel(epo, mnt, {'Cz','PO7'}, ival, defopt_scalp_erp);

plot_scoreMatrix(epo_auc, ival)
ival= procutil_selectTimeIntervals(epo_auc, 'visualize', 1, 'visu_scalps', 1)

%% Classification of ERP data 
% -- classification on spatial features
ival= [0 1000];
epo= proc_segmentation(cnt, mrk, [-200 1000]);
epo= proc_baseline(epo, [-200 0]);
fv= proc_selectIval(epo, ival);
ff= fv;
clear loss
for ii= 1:size(fv.x,1),
  ff.x= fv.x(ii,:,:);
  loss(ii)= crossvalidation(ff, @train_RLDAshrink, 'sampleFcn', {@sample_KFold, 5}, 'LossFcn',@loss_rocArea);
end

clf;
acc= 100-100*loss;
plot(fv.t, acc);


% -- classification on temporal features
fv= proc_selectIval(epo, [0 800]);
ff= fv;
clear loss
for ii= 1:size(fv.x,2),
  ff.x= fv.x(:,ii,:);
  loss(ii)= crossvalidation(ff, @train_RLDAshrink, 'sampleFcn', {@sample_KFold, 5}, 'LossFcn',@loss_rocArea);
end
acc= 100-100*loss;
plot_scalp(mnt, acc, 'CLim','range', 'Colormap', cmap_whitered(31));



% -- classification on spatio-temporal features
ival= [150:50:700; 200:50:750]';
fv= proc_jumpingMeans(epo, ival);
loss_spatioTemp = crossvalidation(fv, @train_RLDAshrink, 'sampleFcn', {@sample_KFold, 5}, 'LossFcn',@loss_rocArea);

%with interval selection based on heuristics
%epo_auc= proc_aucValues(epo);
%ival= select_time_intervals(epo_auc, 'visualize', 1, 'visu_scalps', 1, ...
%                            'nIvals',5);
%fv= proc_jumpingMeans(epo, ival);
%loss_spatioTemp = crossvalidation(fv, @train_RLDAshrink, 'sampleFcn', {@sample_KFold, 5}, 'LossFcn',@loss_rocArea);

% For faster performance, you can switch off type-checking and the 
% history for validation.
tcstate= bbci_typechecking('off');
BTB.History= 0;
% xvalidation(...)
% Put typechecking back in the original state:
bbci_typechecking(tcstate);