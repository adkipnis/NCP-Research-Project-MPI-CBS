function [AUC, AUCtr]= crossvalidation_AK(fv, cvopt, varargin)
%CROSSVALIDATION_AK - Perform cross-validation
%
%
%Arguments:
%  fv -     Struct of feature vectors with data in field '.x' and labels in
%           field '.y'. FV.x may have more than two dimensions. The last
%           dimension is assumed to index samples. The labels FV.y must have
%           the format DOUBLE with size nClasses x nSamples.
%  dec -    Decoding scheme:
%              (0) Overall: concatenate all timepoints of all channels, find weights for T*C
%              (1) Spatial: average all timepoints, find weights for all channels
%              (2) Temporal: separately per channel, find weights for all timepoints 
%              (3) Spatio-temporal: separately per timepoint, find weights for all channels 
%  rep -    Repetitions of procedure
%  perm -   0: normal procedure; 1: permute labels
%  OPT -    Struct or property/value list of optional properties:
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
%Returns:
%  AUC -       Area under the curve for the training set in the format of the specified decoding scheme
%              AUC can indicate overfit when compared to AUCtr
%  AUCtr -     AUC for training set, can indicate underfit
%
% 2014-02 Benjamin Blankertz
% Comments and changes: 2019-09 Alexander Kipnis
fprintf('\n#####################\n# Starting decoding #\n#####################\n')
dec = cvopt.dec; rep = cvopt.rep; perm = cvopt.perm; 


% check decoding scheme
if ~ismember(dec, [0 1 2 3])
  fprintf(2, '\nError: Please specify the decoding scheme (dec = 0, 1, 2, 3). \n')
  return
end

% check if we are doing permutation testing
if perm == 1
  fv_backup = fv;
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
if ismember(dec, [1 3])
    varargin{1,7}.train = {varargin{1,7}.train{1}};
    varargin{1,7}.apply = {varargin{1,7}.apply{1}};
    fprintf(2,'Warning: Disabling proc_variance, as the chosen decoding scheme disallows the use of proc_variance\n(otherwise you loose necessary temporal information).\n')
elseif size(fv.x, 1) == 1
    varargin{1,7}.train = {varargin{1,7}.train{1}};
    varargin{1,7}.apply = {varargin{1,7}.apply{1}};
    fprintf(2,'Warning: Disabling proc_variance, as the input data format disallows the use of proc_variance\n(variance is undefined for n=1 datapoints per channel).\n')
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
applyFcn= misc_getApplyFunc(opt.ClassifierFcn);
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

%% start crossvalidation
for rr= 1:rep, %used to be length(divTr)
    
    if rep > 1, fprintf('\nRepetition %d:\n', rr), end
    
    %permute labels if instructed
    if perm == 1
       fv = fv_backup;
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
  [fold_loss, fold_lossTr]= decode_bcci(fv, dec, idxTr, idxTe, ff, rr, ...
        fold_loss, fold_lossTr, trainFcn, trainPar, applyFcn, lossFcn, lossPar, opt);
    
  end
  
  % calculate crossvalidation loss
  xv_loss(rr,:)= mean(fold_loss);
  xv_lossTr(rr,:)= mean(fold_lossTr);

end

% final outputs
%if nOutDim==1,
%  cfy_out= reshape(cfy_out, [length(divTe), size(fv.y,2)]);
%end
loss= mean(xv_loss,1);
lossSem= std(xv_loss,0,1)/sqrt(size(xv_loss,1));
lossTr= mean(xv_lossTr,1);
lossTrSem= std(xv_lossTr,0,1)/sqrt(size(xv_lossTr,1));
AUC = 100-100*loss; AUCtr = 100-100*lossTr;
