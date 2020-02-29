function [Acc, Acc_tr, p, stat, A]= crossvalidation_AK(fv, cvopt, varargin)
%CROSSVALIDATION_AK - Perform cross-validation
%
%
%Arguments:
%  fv -     Structure of feature vectors with data in field '.x' and labels in
%           field '.y'. FV.x may have more than two dimensions. The last
%           dimension is assumed to index samples. The labels FV.y must have
%           the format DOUBLE with size nClasses x nSamples.
%
% varargin - Arguments that will be converted to a structure or property/value list of optional properties:
%   'SampleFcn': Function handle of sampling function, see functions
%           sample_*, or CELL providing also parameters of the samling
%           function), default @sample_KFold
%   'LossFcn': Function handle of loss function, CELL involving parameters 
%           {@FCN, PARAM1, PARAM2, ...}, or CELL array of function 
%           handles for multiple loss statistics (e.g. {@loss_0_1,
%           @loss_rocArea} ) - no parameters possible.
%   'ClassifierFcn': as direct input argument CLASSY, see above;
%           default @train_RLDAshrink
%   'Proc': Struct with fields 'train' and 'apply'. Each of those is a CELL
%           specifying a processing chain. See the example
%           demo_validation_csp to learn about this feature.
%
%  cvopt -  Structure containing options that further customize the crossvalidation procedure
%  'dec'       -    Decoding scheme:
%                   (0) Overall: concatenate all timepoints of all channels, find weights for T*C
%                   (1) Spatial: average all timepoints, find weights for all channels
%                   (2) Temporal: separately per channel, find weights for all timepoints 
%                   (3) Spatio-temporal: separately per timepoint, find weights for all channels 
%  'rep'       -    Repetitions of procedure
%  'perm'      -    0: normal procedure; 1: permute labels
%  'covariate' -    1 x Tr feature vector (typically another class index) for which each LDA training will
%                   find an optimal weight, the weight and the feature will be removed
%                   before classification. This is analogous to partializing the covariate out.
%
%Returns:
%  Acc      -     Accuracy (formerly Area under the curve) for the training set in the format of the specified decoding scheme
%                 Acc can indicate overfit when compared to Acc_tr
%  Acc_tr   -     Accuracy for training set, can indicate underfit
%  p        -     Cumulative probability density of empirical Acc and Accs larger than it, given a baseline decoding accuracy of 50%
%  stat     -     Structure with fields bcdf_x (foldwise correct classifications) and bcdf_n (foldwise classification attempts = testset size)
%  A        -     If a spatial filter is specified in proc.train, A is the mean activation pattern matrix over folds over repetitions.
%
% 2014-02 Benjamin Blankertz
% Comments and changes: 2019-09 Alexander Kipnis
fprintf('\n------------------------\nStart decoding analysis \n------------------------\n');
dec = cvopt.dec; rep = cvopt.rep; perm = cvopt.perm; 
fv_backup = fv; A = [];
undersample(fv_backup, 1); %to tell if any trials will be removed

% check decoding scheme
if ~ismember(dec, [0 1 2 3])
  fprintf(2, '\nError: Please specify the decoding scheme (dec = 0, 1, 2, 3). \n')
  return
end


% check if we are doing permutation testing
if perm == 1
  fprintf('\nThe labels of this dataset will be permuted in each repetition. \n')
end


props = {'SampleFcn'       {@sample_KFold, [10 10]}   '!FUNC|CELL'
         'LossFcn'         @loss_0_1                  '!FUNC|CELL'
         'ClassifierFcn'   @train_RLDAshrink          '!FUNC|CELL'
         'Proc'            []                         'STRUCT'
         };

if nargin==0;
  loss= props; return
end

%when we only supply one data point per channel per trial, proc_variance
%and proc_logarithm do not work :)
if ismember(dec, [1 3]) & length(varargin)> 5
    if contains(char(varargin{1,6}),'Proc') & contains(char(varargin{1,7}.train{2}),'proc_variance')
    varargin{1,7}.train = {varargin{1,7}.train{1}};
    varargin{1,7}.apply = {varargin{1,7}.apply{1}};
    fprintf(2,'Warning: Disabling proc_variance, as the chosen decoding scheme disallows the use of proc_variance\n(otherwise you loose necessary temporal information).\n')
    end
elseif size(fv.x, 1) == 1 & contains(char(varargin{1,6}),'Proc')
    if contains(char(varargin{1,6}),'Proc') & contains(char(varargin{1,7}.train{2}),'proc_variance')
    varargin{1,7}.train = {varargin{1,7}.train{1}};
    varargin{1,7}.apply = {varargin{1,7}.apply{1}};
    fprintf(2,'Warning: Disabling proc_variance, as the input data format disallows the use of proc_variance\n(variance is undefined for n=1 datapoints per channel).\n')
    end
end

% convert inputs in varargin to opt structure
if misc_isproplist(varargin{1}),
  opt= opt_proplistToStruct(varargin{:});
else
  opt= opt_proplistToStruct(varargin{2:end});
  opt.ClassifierFcn= varargin{1};
end

% compare opt structure to defaults (props) and fill in undefined parameters
[opt,isdefault] = opt_setDefaults(opt, props, 1);
misc_checkType(fv, 'STRUCT(x y)');
misc_checkType(fv.x, 'DOUBLE[2- 1]|DOUBLE[1- 2-]|DOUBLE[- - -]', 'fv.x');

% set proc field in opt (important for crossvalidation)
opt.Proc= xvalutil_procSetDefault(opt.Proc);

% set function handles
[trainFcn, trainPar]= misc_getFuncParam(opt.ClassifierFcn);
if contains(char(opt.ClassifierFcn{1,1}),'svm')
  applyFcn =  @predict;
else
  applyFcn = misc_getApplyFunc(opt.ClassifierFcn);  
end

[sampleFcn, samplePar]= misc_getFuncParam(opt.SampleFcn);
[lossFcn, lossPar]= misc_getFuncParam(opt.LossFcn);

% initialize crossvalidation loss
%if isempty(lossPar)||~isa(lossPar{1}, 'function_handle')
%    xv_loss= zeros(length(divTr), 1);
%    xv_lossTr= zeros(length(divTr), 1);
%else
%    xv_loss= zeros(length(divTr), size(lossPar,2)+1);
%    xv_lossTr= zeros(length(divTr), size(lossPar,2)+1);
%end

% how many classes are to be classified?
nOutDim= size(fv.y,1);
if nOutDim==2,
  nOutDim= 1;
end
%cfy_out= NaN*zeros(nOutDim, length(divTe), size(fv.y,2));

% Optionally append each trial with covariate feature
if isfield(cvopt, 'covariate')
   fv_backup.x = cat(1,fv.x, cvopt.covariate); %appends each trial
   opt.covariate = 1; %important for decode_bcci
   fprintf('The provided covariate will be used for training, but it and its weight will be removed.\n')
end

%% start crossvalidation
for rr= 1:rep, %used to be length(divTr)
    
    if rep > 1, fprintf('\nRepetition %d:\n', rr), end
    
    %undersample
    if cvopt.us == 1
       fv = undersample(fv_backup, 0); % have an equal amount of samples for each class, silent output
    end
    
    %permute labels if instructed
    if perm == 1
       if cvopt.us == 0, fv = fv_backup; end
       idx = randperm(size(fv.y, 2));
       fv.y = fv.y(:,idx); %permute labels
    end
    
    %index training and test sets for each fold according to sampling function
    [divTr, divTe]= sampleFcn(fv.y, samplePar{:});  
    nFolds= length(divTr{1});
   
    % initialize fold loss
  if isempty(lossPar)||~isa(lossPar{1}, 'function_handle')
     fold_loss= zeros(nFolds, 1);
     fold_lossTr= zeros(nFolds, 1);
  else
     fold_loss= zeros(nFolds, size(lossPar,2)+1);
     fold_lossTr= zeros(nFolds, size(lossPar,2)+1);
  end
  
  % begin k folds
  for ff= 1:nFolds,
    fprintf('Fold %d... | ', ff)
    idxTr= divTr{1}{ff}; % access indices defined above
    idxTe= divTe{1}{ff};
    
  % decoding within fold, according to prespecified decoding scheme 
  [fold_loss, fold_lossTr, A_fold]= decode_bcci(fv, dec, idxTr, idxTe, ff, rr, ...
        fold_loss, fold_lossTr, trainFcn, trainPar, applyFcn, lossFcn, lossPar, opt);
  A = cat(3, A, A_fold); %save activation pattern for later averaging
    
  end
  
  % calculate x and n for binocdf
  for i=(rr-1)*nFolds+1:rr*nFolds
      j = mod(i,10); if j==0, j=10; end %enables stacking of losses of kfolds over repetition
      bcdf_n(i,1,1:size(fold_loss, 3)) = size(divTe{1,1}{1,j}, 2); %amount of to be classified trials - the sum of this vector should be = size(fv.x, 3)
      bcdf_x(i,1,:) = round(bcdf_n(i,1)*(1-fold_loss(j,1,:)), 0); %amount of correctly classified trials NOTE: this only works if the first lossfcn is computing accuracy!
  end 
  
  % calculate crossvalidation loss (mean loss over all folds)
  if dec >= 2, %for temporal and spatial decoding over time
    xv_loss(rr,1:size(fold_loss,2),:)= mean(fold_loss,1); %otherwise we get a dimension mismatch 
    xv_lossTr(rr,1:size(fold_lossTr,2),:)= mean(fold_lossTr,1);
  else
    xv_loss(rr,1:size(fold_loss,2))= mean(fold_loss,1);
    xv_lossTr(rr,1:size(fold_lossTr,2))= mean(fold_lossTr,1);
  end
  
  
end

%"Significance test" (over all folds and repetitions) with binomial cumulative distribution function 
p = 1 - binocdf(sum(bcdf_x),sum(bcdf_n),0.5); %probability of x or more correct classifications out of n classification attempts, given a 50% chance of correct classification (i.e., under the null distribution)
stat.bcdf_x = bcdf_x; stat.bcdf_n = bcdf_n;
%like a one-tailed t-test against null-distribution

% average activation patterns 
A = mean(A,3);

% final outputs
%if nOutDim==1,
%  cfy_out= reshape(cfy_out, [length(divTe), size(fv.y,2)]);
%end
if perm == 0
 loss= mean(xv_loss,1);
 lossSem= std(xv_loss,0,1)/sqrt(size(xv_loss,1));
 lossTr= mean(xv_lossTr,1);
 lossTrSem= std(xv_lossTr,0,1)/sqrt(size(xv_lossTr,1));

elseif perm == 1
 loss = xv_loss; % permute(xv_loss, [3 2 1]);
 lossTr = xv_lossTr; %permute(xv_lossTr, [1 2 3]);   
end
Acc = 100-100*loss; Acc_tr = 100-100*lossTr;
