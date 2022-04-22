close all;
clear;
clc;

load('COIL20.mat');
X = X';
K = max(y);
gnd = y';
n = size(X, 2);

cluster_data = cell(1, K);
class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(gnd == idx));
end

lambdas =[1e-4];
betas = [1e-2];
mus = [0.1];
rhos = [9];
gammas = [1.05];

for lmd_idx = 1 : length(lambdas)
    lambda = lambdas(lmd_idx);
    for beta_idx = 1 : length(betas)
        beta = betas(beta_idx);                
        for mu_idx = 1 : length(mus)
            mu = mus(mu_idx);            
            for rho_idx = 1 : length(rhos)
                rho = rhos(rho_idx);
                for gamma_idx = 1 : length(gammas)
                    gamma = gammas(gamma_idx);
                    tic;
                    [Z, D, labels, iter, ~, ~] = tssrc(normc(X), lambda, beta, mu, rho, gamma, K);
                    time_cost = toc;
                    acc = accuracy(gnd', labels);                  
                    for pos_idx =  1 : K
                        cluster_data(1, pos_idx) = { gnd(labels == pos_idx) };
                    end
                    [nmi, fmeasure] = calculate_results(class_labels, cluster_data);
                    disp([lambda, beta, mu, rho, gamma, acc,  nmi, fmeasure, iter]);
                    dlmwrite('trsc_coil_clustering_result.txt', [lambda, beta, mu, rho, gamma, acc,  nmi, fmeasure, iter, time_cost], '-append', 'delimiter', '\t', 'newline', 'pc');
                 end
            end
        end
    end
end



