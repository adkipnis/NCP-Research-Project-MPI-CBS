function [Pval]= perm_CSP_eigen(fv, OPTcsp, OPTp)
%perm_CSP_eigen - Cluster-based permutation test for eigenvalues produced by CSP
%
%
%Arguments:
%  fv  -       EEG structure in BBCI format
%  OPTcsp -    Structure with settings for CSP (see proc_csp.m)
%  OPTp -      Structure with settings for permutation testing
%  'np'   -    Number of permutations
%
%Returns:
%  CSP -       EEG structure in BBCI format after CSP has been applied;
%              also containts the permutation test p-values in CSP.Pval
%

% 2019-09 AK, mostly copied from the SPOC function!


%extract parameters
n_bootstrapping_iterations = OPTp.np; y = fv.y(1,:);


% Empirical CSP
[DAT, CSP_W, CSP_A, EigVal] = proc_csp(fv, OPTcsp);

%% Bootstrapping procedure
n_ev = length(EigVal);
p_values_lambda = inf(1,n_ev);
p_values_r = inf(1,n_ev);
%y_amps = [];
fprintf('0%% completed... \n')

if n_bootstrapping_iterations > 0
    lambda_samples = zeros(1, n_bootstrapping_iterations);
    for k=1:n_bootstrapping_iterations
        
        percent = k/n_bootstrapping_iterations * 100;
        if mod(percent,10)==0, fprintf('%d%% completed... \n', percent), end
        % shuffle the target function
        %[y_shuffled, y_amps] = random_phase_surrogate(y, 'z_amps', y_amps); % WIP!
        y_shuffled = y(randperm(length(y))); y_shuffled(2,:) = abs(y_shuffled-1); % shuffled label vector
        fp = fv; fp.y = y_shuffled;
        
        % re-compute CSP
        [~, ~, ~, EigVal_perm] = proc_csp(fp, OPTcsp);
        
        % store permutation lambdas 
        lambda_samples(k) = max(abs(EigVal_perm));
    end
    
    % compute bootstrapped p-values
    for n=1:n_ev
        p_values_lambda(n) = sum(abs(lambda_samples(:))>=abs(EigVal(n)))/n_bootstrapping_iterations;
    end
end



%% store variables
%CSP.DAT = DAT; CSP.W = CSP_W; CSP.A = CSP_A; CSP.EigVal = EigVal; CSP.Pval = p_values_lambda; 

end