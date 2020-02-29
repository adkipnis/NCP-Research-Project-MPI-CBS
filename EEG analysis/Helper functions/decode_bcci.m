function [fold_loss, fold_lossTr, A_fold]= decode_bcci(fv, dec, idxTr, idxTe, ff, rr, ...
        fold_loss, fold_lossTr, trainFcn, trainPar, applyFcn, lossFcn, lossPar, opt)
%DECODE_BCCI
% All arguments are directly taken from the crossvalidation_AK function
%
%     dec -    Decoding scheme:
%              (0) Overall: concatenate all timepoints of all channels, find weights for T*C
%              (1) Spatial: average all timepoints, find weights for all channels
%              (2) Temporal: separately per channel, find weights for all timepoints 
%              (3) Spatio-temporal: separately per timepoint, find weights for all channels 
% opt.covariate - The existence of this field tweaks the LDA classification to include the covariate in fv.x for training
%                 but exclude it and its weights for testing.
%                 ### ONLY CODED FOR dec = 0 !
% 
% 
%
% 2019-09 AK
A_fold = [];
%% decoding with overall features: concatenate all channels, find weights for all timepoints
if dec == 0 
    % sample training set
    fvTr= proc_selectSamples(fv, idxTr);
    
    %separate covariate (otherwise it affects CSP)
    if isfield (opt, 'covariate')
        covariate = fvTr.x(end,:,:); %copy to workspace
        fvTr.x(end,:,:) = []; %delete from data array
    end
    
    % specific options for crossvalidation with different training and test
    % procedures
    
    
    if ~isempty(opt.Proc),      
      [fvTr, memo]= xvalutil_proc(fvTr, opt.Proc.train); % e.g. apply CSP to training set
      if contains(char(opt.Proc.train{1,1}{1,2}), 'csp')
          A_fold = fvTr.A; %save activation pattern
      end
      
    end
    
    % train classifier on train set    
    %add covariate back in ### Warning: this only works if the first dimension of fvTr.x has the size 1!
    if isfield (opt, 'covariate')
        fvTr.x = cat(2,fvTr.x, covariate(:,1,:)); %appends each trial
    end
    
    xsz= size(fvTr.x);
    fvsz= [prod(xsz(1:end-1)) xsz(end)];
    if contains(char(opt.ClassifierFcn{1,1}),'svm')
        C= trainFcn(reshape(fvTr.x,fvsz)', fvTr.y(2,:)', trainPar{:}); %right format for libsvm
    else
        C= trainFcn(reshape(fvTr.x,fvsz), fvTr.y, trainPar{:});
    end   
    
    %delete weight for covariate, remove feature from test set and reset fvsz
    if isfield (opt, 'covariate')
        C.w(end,:) = []; %delete weight
        fvTr.x(:,end,:) = []; % delete covariate
        xsz= size(fvTr.x);
        fvsz= [prod(xsz(1:end-1)) xsz(end)]; % reset fvsz
    end
    
    % sample test set
    fvTe= proc_selectSamples(fv, idxTe);
    %separate covariate (otherwise it affects CSP)
    if isfield (opt, 'covariate')
        covariate = fvTe.x(end,:,:); %copy to workspace
        fvTe.x(end,:,:) = []; %delete from data array
    end
    
    
    % proc apply onto test set (e.g. same spatial filter as found in training set)
    if ~isempty(opt.Proc),
      fvTe= xvalutil_proc(fvTe, opt.Proc.apply, memo);
    end
    
    % test classifier on test set
    xsz= size(fvTe.x);
    if length(xsz) == 2
       xsz = [xsz 1]; %if leave one out approach is chosen, the third dimension of x is 1 and is not accounted for
    end
    
    if contains(char(opt.ClassifierFcn{1,1}),'svm')
        out= [applyFcn(C, reshape(fvTe.x, [prod(xsz(1:end-1)) xsz(end)])')]';
        outTr= [applyFcn(C, reshape(fvTr.x, fvsz)')]'; 
        out = out-(out==0); %every 0 prediction is transformed to -1. The classification performance is measured by taking the sign of the prediction, hence this step is necessary.
        outTr = outTr-(outTr==0);
    else
        out= applyFcn(C, reshape(fvTe.x, [prod(xsz(1:end-1)) xsz(end)]));
        % ...and on training set
        outTr= applyFcn(C, reshape(fvTr.x, fvsz)); 
    end   
    
    cfy_out(:,rr,idxTe)= out; %save predictions
    
    
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
      if contains(char(opt.Proc.train{1,1}{1,2}), 'csp')
          A_fold = fvTr.A; %save activation pattern
      end
    end
    
    
    
    % train classifier on train set
    
    xsz= size(fvTr.x);
    fvsz= [prod(xsz(1:end-1)) xsz(end)];
    if contains(char(opt.ClassifierFcn{1,1}),'svm')
        C= trainFcn(reshape(fvTr.x,fvsz)', fvTr.y(2,:)', trainPar{:}); %right format for libsvm
    else
        C= trainFcn(reshape(fvTr.x,fvsz), fvTr.y, trainPar{:});
    end   
    
    
    
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
      
      [fvTr, memo]= xvalutil_proc(fvTr, opt.Proc.train); % e.g. apply CSP to training set
      if contains(char(opt.Proc.train{1,1}{1,2}), 'csp')
          A_fold = fvTr.A; %save activation pattern
      end
    end
    
    
    %define channel numbers and training subset
    nChan = size(fvTr.x, 2);
    fvTr_c = fvTr; fvTe_c = fvTe;
    
    for chan = 1:nChan
       
       fvTr_c.x =  fvTr.x(:,chan,:);
       xsz= size(fvTr_c.x);
       fvsz= [prod(xsz(1:end-1)) xsz(end)];
       
       if contains(char(opt.ClassifierFcn{1,1}),'svm')
           C= trainFcn(reshape(fvTr_c.x,fvsz)', fvTr_c.y(2,:)', trainPar{:}); %right format for libsvm
       else
           C= trainFcn(reshape(fvTr_c.x,fvsz), fvTr_c.y, trainPar{:});
       end   
    
       
       % define test subset and apply a separating hyperplane
       fvTe_c.x =  fvTe.x(:,chan,:);
       xsz= size(fvTe_c.x);
       if length(xsz) == 2
           xsz = [xsz 1]; %if leave one out approach is chosen, the third dimension of x is 1 and is not accounted for
       end
       out (chan,:) = applyFcn(C, reshape(fvTe_c.x, [prod(xsz(1:end-1)) xsz(end)])); %output for each trial in the test set   
       cfy_out(:,chan,idxTe)= out(chan,:);    
       outTr(chan,:)= applyFcn(C, reshape(fvTr_c.x, fvsz)); %output for each trial in the training set
       
       % calculate fold loss
       if isempty(lossPar)||~isa(lossPar{1}, 'function_handle')              
            fold_loss(ff,:, chan)= mean(lossFcn(fvTe.y, out(chan,:), lossPar{:}));
            fold_lossTr(ff,:, chan)= mean(lossFcn(fvTr.y, outTr(chan,:), lossPar{:}));
       else
       losstmp=[];
       losstmpTr=[];
            for ii=1:size(lossPar,2)
                losstmp=[losstmp mean(lossPar{ii}(fvTe.y, out(chan,:)))];
                losstmpTr=[losstmpTr mean(lossPar{ii}(fvTe.y, outTr(chan,:)))];
            end
       fold_loss(ff,:, chan)= [mean(lossFcn(fvTe.y, out(chan,:))) losstmp];
       fold_lossTr(ff,:,chan)= [mean(lossFcn(fvTr.y, outTr(chan,:))) losstmpTr];        
       end
       
       %fold_loss(ff,:, chan)= mean(lossFcn(fvTe.y, out(chan,:), lossPar{:}));
       %fold_lossTr(ff,:, chan)= mean(lossFcn(fvTr.y, outTr(chan,:), lossPar{:}));

    end

%% Decoding with spatial features over time: separately per timepoint, find weights for all channels    
elseif dec == 3 
   
   % define training set and train the specified classifier
    fvTr= proc_selectSamples(fv, idxTr); 
      
    if ~isempty(opt.Proc),
      
      [fvTr, memo]= xvalutil_proc(fvTr, opt.Proc.train); % e.g. apply CSP to training set
      if contains(char(opt.Proc.train{1,1}{1,2}), 'csp')
          A_fold = fvTr.A; %save activation pattern
      end
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
       
       if contains(char(opt.ClassifierFcn{1,1}),'svm')
           C= trainFcn(reshape(fvTr_s.x,fvsz)', fvTr_s.y(2,:)', trainPar{:}); %right format for libsvm
       else
           C= trainFcn(reshape(fvTr_s.x,fvsz), fvTr_s.y, trainPar{:});
       end
       
       
       
       % define test subset and apply a separating hyperplane
       fvTe_s.x =  fvTe.x(sam,:,:);
       xsz= size(fvTe_s.x);
       
       if length(xsz) == 2
           xsz = [xsz 1]; %if leave one out approach is chosen, the third dimension of x is 1 and is not accounted for
       end
       
       out (sam,:) = applyFcn(C, reshape(fvTe_s.x, [prod(xsz(1:end-1)) xsz(end)])); %output for each trial in the test set   
       cfy_out(:,sam,idxTe)= out(sam,:);    
       outTr(sam,:)= applyFcn(C, reshape(fvTr_s.x, fvsz)); %output for each trial in the training set

       
       % calculate fold loss
       if isempty(lossPar)||~isa(lossPar{1}, 'function_handle')
            fold_loss(ff,1, sam)= mean(lossFcn(fvTe_s.y, out(sam,:), lossPar{:}));                  
            fold_lossTr(ff,1, sam)= mean(lossFcn(fvTr_s.y, outTr(sam,:), lossPar{:}));
       else
       losstmp=[];
       losstmpTr=[];
            for ii=1:size(lossPar,2)
                losstmp=[losstmp mean(lossPar{ii}(fvTe_s.y, out(sam,:)))];
                losstmpTr=[losstmpTr mean(lossPar{ii}(fvTr_s.y, outTr(sam,:)))];
            end
       fold_loss(ff,:, sam)= [mean(lossFcn(fvTe_s.y, out(sam,:))) losstmp];
       fold_lossTr(ff,:,sam)= [mean(lossFcn(fvTr_s.y, outTr(sam,:))) losstmpTr];        
       end     
       
       %fold_loss(ff, sam)= mean(lossFcn(fvTe_s.y, out(sam,:), lossPar{:}));                  
       %fold_lossTr(ff, sam)= mean(lossFcn(fvTr_s.y, outTr(sam,:), lossPar{:}));
   end
end



end
