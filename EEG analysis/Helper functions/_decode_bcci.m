function [fold_loss, fold_lossTr]= decode_bcci(fv, dec, idxTr, idxTe, ff, rr, ...
        fold_loss, fold_lossTr, trainFcn, trainPar, applyFcn, lossFcn, lossPar, opt)
%DECODE_BCCI
% All arguments are directly taken from the crossvalidation_AK function
%
%     dec -    Decoding scheme (WIP):
%              (0) Overall: concatenate all timepoints of all channels, find weights for T*C
%              (1) Spatial: average all timepoints, find weights for all channels
%              (2) Temporal: separately per channel, find weights for all timepoints 
%              (3) Spatio-temporal: separately per timepoint, find weights for all channels 
% 
%
% 2019-09 AK
%% decoding with overall features: concatenate all channels, find weights for all timepoints
if dec == 0 
    % sample training set
    fvTr= proc_selectSamples(fv, idxTr);
    
    % specific options for crossvalidation with different training and test
    % procedures
    if ~isempty(opt.Proc),
      [fvTr, memo]= xvalutil_proc(fvTr, opt.Proc.train); % e.g. apply CSP to training set
    end
    
    % train classifier on train set
    xsz= size(fvTr.x);
    fvsz= [prod(xsz(1:end-1)) xsz(end)];
    C= trainFcn(reshape(fvTr.x,fvsz), fvTr.y, trainPar{:});
    
    % sample test set
    fvTe= proc_selectSamples(fv, idxTe);
    
    % proc apply onto test set (e.g. same spatial filter as found in training set)
    if ~isempty(opt.Proc),
      fvTe= xvalutil_proc(fvTe, opt.Proc.apply, memo);
    end
    
    % test classifier on test set
    xsz= size(fvTe.x);
    if length(xsz) == 2
       xsz = [xsz 1]; %if leave one out approach is chosen, the third dimension of x is 1 and is not accounted for
    end
    out= applyFcn(C, reshape(fvTe.x, [prod(xsz(1:end-1)) xsz(end)]));
    cfy_out(:,rr,idxTe)= out; %save predictions
    % ...and on training set
    outTr= applyFcn(C, reshape(fvTr.x, fvsz)); 
    
    % calculate fold loss
    if isempty(lossPar)||~isa(lossPar{1}, 'function_handle')
        fold_loss(ff)= mean(lossFcn(fvTe.y, out, lossPar{:}));
        fold_lossTr(ff)= mean(lossFcn(fvTr.y, outTr, lossPar{:}));
    else
        losstmp=[];
        losstmpTr=[];
        for ii=1:size(lossPar,2)
            losstmp=[losstmp mean(lossPar{ii}(fvTe.y, out))];
            losstmpTr=[losstmpTr mean(lossPar{ii}(fvTr.y, outTr))];
        end
        fold_loss(ff,:)= [mean(lossFcn(fvTe.y, out)) losstmp];
        fold_lossTr(ff,:)= [mean(lossFcn(fvTr.y, outTr)) losstmpTr];        
    end

%% decoding with spatial features: average all timepoints, find weights for all channels
elseif dec == 1
     % sample training set
    fvTr= proc_selectSamples(fv, idxTr); 
    fvTr.x = mean(fvTr.x, 1); %average all timepoints
    % specific options for crossvalidation with different training and test
    % procedures
    if ~isempty(opt.Proc),
      [fvTr, memo]= xvalutil_proc(fvTr, opt.Proc.train); % e.g. apply CSP to training set
    end
    
    
    
    % train classifier on train set
    xsz= size(fvTr.x);
    fvsz= [prod(xsz(1:end-1)) xsz(end)];
    C= trainFcn(reshape(fvTr.x,fvsz), fvTr.y, trainPar{:});
    
    % sample test set
    fvTe= proc_selectSamples(fv, idxTe);
    fvTe.x = mean(fvTe.x, 1); %average all timepoints
    % proc apply onto test set (e.g. same spatial filter as found in training set)
    if ~isempty(opt.Proc),
      fvTe= xvalutil_proc(fvTe, opt.Proc.apply, memo);
    end
    
   
    
    % test classifier on test set
    xsz= size(fvTe.x);
    if length(xsz) == 2
        xsz = [xsz 1]; %if leave one out approach is chosen, the third dimension of x is 1 and is not accounted for
    end
    out= applyFcn(C, reshape(fvTe.x, [prod(xsz(1:end-1)) xsz(end)]));
    cfy_out(:,rr,idxTe)= out; %save predictions
    % ...and on training set
    outTr= applyFcn(C, reshape(fvTr.x, fvsz)); 
    
    % calculate fold loss
    if isempty(lossPar)||~isa(lossPar{1}, 'function_handle')
        fold_loss(ff)= mean(lossFcn(fvTe.y, out, lossPar{:}));
        fold_lossTr(ff)= mean(lossFcn(fvTr.y, outTr, lossPar{:}));
    else
        losstmp=[];
        losstmpTr=[];
        for ii=1:size(lossPar,2)
            losstmp=[losstmp mean(lossPar{ii}(fvTe.y, out))];
            losstmpTr=[losstmpTr mean(lossPar{ii}(fvTr.y, outTr))];
        end
        fold_loss(ff,:)= [mean(lossFcn(fvTe.y, out)) losstmp];
        fold_lossTr(ff,:)= [mean(lossFcn(fvTr.y, outTr)) losstmpTr];        
    end

%% decoding with temporal features: separately per channel, find weights for all timepoints     
elseif dec == 2 
   
    % define training set and train the specified classifier
    fvTr= proc_selectSamples(fv, idxTr); 
      
    if ~isempty(opt.Proc),
       [fvTr, memo]= xvalutil_proc(fvTr, opt.Proc.train); % e.g. apply CSP to training set
    end
    
    % sample test set
    fvTe= proc_selectSamples(fv, idxTe);
    % proc apply onto test set (e.g. same spatial filter as found in training set)
    if ~isempty(opt.Proc),
      fvTe= xvalutil_proc(fvTe, opt.Proc.apply, memo);
    end
    
    
    %define channel numbers and training subset
    nChan = size(fvTr.x, 2);
    fvTr_c = fvTr; fvTe_c = fvTe;
    
    for chan = 1:nChan
       
       fvTr_c.x =  fvTr.x(:,chan,:);
       xsz= size(fvTr_c.x);
       fvsz= [prod(xsz(1:end-1)) xsz(end)];
       C= trainFcn(reshape(fvTr_c.x,fvsz), fvTr_c.y, trainPar{:});

       % define test subset and apply a separating hyperplane
       fvTe_c.x =  fvTe.x(:,chan,:);
       xsz= size(fvTe_c.x);
       if length(xsz) == 2
           xsz = [xsz 1]; %if leave one out approach is chosen, the third dimension of x is 1 and is not accounted for
       end
       out (chan,:) = applyFcn(C, reshape(fvTe_c.x, [prod(xsz(1:end-1)) xsz(end)])); %output for each trial in the test set   
       cfy_out(:,chan,idxTe)= out(chan,:);    
       outTr(chan,:)= applyFcn(C, reshape(fvTr_c.x, fvsz)); %output for each trial in the training set
       
       fold_loss(ff, chan)= mean(lossFcn(fvTe.y, out(chan,:), lossPar{:}));
       fold_lossTr(ff, chan)= mean(lossFcn(fvTr.y, outTr(chan,:), lossPar{:}));
    end

%% Decoding with spatial features over time: separately per timepoint, find weights for all channels    
elseif dec == 3 
   
   % define training set and train the specified classifier
    fvTr= proc_selectSamples(fv, idxTr); 
      
    if ~isempty(opt.Proc),
       [fvTr, memo]= xvalutil_proc(fvTr, opt.Proc.train); % e.g. apply CSP to training set
    end
    
    % sample test set
    fvTe= proc_selectSamples(fv, idxTe);
    % proc apply onto test set (e.g. same spatial filter as found in training set)
    if ~isempty(opt.Proc),
      fvTe= xvalutil_proc(fvTe, opt.Proc.apply, memo);
    end
     
    
   %define trial duration 
   nSample = size(fvTr.x, 1);
   fvTr_s =  fvTr; fvTe_s =  fvTe;
   
   for sam = 1:nSample
      
       % define training subset and train the specified classifier
       fvTr_s.x =  fvTr.x(sam,:,:);
       xsz= size(fvTr_s.x);
       fvsz= [prod(xsz(1:end-1)) xsz(end)];
       C= trainFcn(reshape(fvTr_s.x,fvsz), fvTr_s.y, trainPar{:});

       % define test subset and apply a separating hyperplane
       fvTe_s.x =  fvTe.x(sam,:,:);
       xsz= size(fvTe_s.x);
       
       if length(xsz) == 2
           xsz = [xsz 1]; %if leave one out approach is chosen, the third dimension of x is 1 and is not accounted for
       end
       
       out (sam,:) = applyFcn(C, reshape(fvTe_s.x, [prod(xsz(1:end-1)) xsz(end)])); %output for each trial in the test set   
       cfy_out(:,sam,idxTe)= out(sam,:);    
       outTr(sam,:)= applyFcn(C, reshape(fvTr_s.x, fvsz)); %output for each trial in the training set

       fold_loss(ff, sam)= mean(lossFcn(fvTe_s.y, out(sam,:), lossPar{:}));
       fold_lossTr(ff, sam)= mean(lossFcn(fvTr_s.y, outTr(sam,:), lossPar{:}));
   end
end 


end
