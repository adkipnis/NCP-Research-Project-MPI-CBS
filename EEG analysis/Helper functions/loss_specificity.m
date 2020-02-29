function loss= loss_specificity(label, out)
%LOSS_specificity - Loss function: prediction error | sample is from class 2 (negative class)
%
%
%Arguments:
% LABEL - matrix of true class labels, size [nClasses nSamples]
% OUT   - matrix (or vector for 2-class problems) of classifier outputs
%
%Returns:
% LOSS  - vector of 0-1 loss values for samples whose true label is class 2 (negative class)
%         i.e., the mean(LOSS) = Correct rejection / (Correct rejection + False alarm)
%
% SEE crossvalidation

% Alexander Kipnis


lind= [1:size(label,1)]*label;
if size(out,1)==1,
  est= 1.5 + 0.5*sign(out);
else
  [dmy, est]= max(out, [], 1);
end

%loss= est~=lind;
[~,negind]= find(lind==2); %indices of trials with negative class
loss= est(negind)~=lind(negind); % loss = missclassified negative class
