function loss= loss_sensitivity(label, out)
%LOSS_sensitivity - Loss function: prediction error | sample is from class 1
%
%
%Arguments:
% LABEL - matrix of true class labels, size [nClasses nSamples]
% OUT   - matrix (or vector for 2-class problems) of classifier outputs
%
%Returns:
% LOSS  - vector of 0-1 loss values for samples whose true label is class 1
%         i.e., the mean(LOSS) = Hit / (Hit+Miss)
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
[~,posind]= find(lind==1); %indices of trials with positive class
loss= est(posind)~=lind(posind); % loss = missclassified positive class
