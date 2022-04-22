function [Z, D, labels, iter, obj_values, z_con_values] = tssrc(X, lambda, beta, mu, rho, gamma, num_clusters)

% default parameters
% rho = 6;
max_mu = 1e6;
tol = 1e-3;
max_iter = 50;

n = size(X, 2);
Z = zeros(n, n);
Y = zeros(n, n);

D = X;
% Dtmp = D;

z_con_values = zeros(1, max_iter);
obj_values = zeros(1, max_iter);

Probs = zeros(n, num_clusters);

T = eye(num_clusters);

Ik = eye(num_clusters);
randorder = randperm(size(Ik,1));
numceil = ceil(n/num_clusters);
largeY = repmat(Ik(randorder,:),numceil,1); %(size(A,1)*M, size(A,2)*N)
probs = largeY(1:n, :); % N*k



% W = constructW_PKN(X{i}, 5, 1);  %   X : d*n  / 5 neighbors /  is symmetric
% D = diag(sum(W));
% L = D-W;
% [F, ~, ~]=eig1(L, num_clusters, 0); 
Q = zeros(n, n);

iter = 0;
obj_tmp = 0;
while iter < max_iter

    
    % update J    
    tmp1 = lambda * (D' * D) + mu * eye(n);
    tmp2 = lambda * (D' * X) + mu * Z - Y - 0.25 * Q;   
    J = normc(tmp1 \ tmp2);
        
    % update Z
    tmp = J + Y/mu;
    thr = sqrt(lambda / mu);
    Z = tmp.*((sign(abs(tmp)-thr)+1)/2);
    ind = abs(abs(tmp)-thr) <= 1e-6;
    Z(ind) = 0;
    Z = Z - diag(diag(Z));
    
    % update D
    A = Z * Z' + beta / lambda * eye(n); 
    B = X * Z';
    for i = 1 : n
        if(A(i, i) ~= 0)
            a = 1.0 / A(i,i) * (B(:,i) - D * A(:, i)) + D(:,i);
            D(:,i) = a / (max( norm(a, 2),1));		
        end
    end

    DD = diag(1./sqrt(sum(Z, 2)+ eps));
    W = DD * Z * DD;
    [U, ~, ~] = svd(W);
    F = U(:, 1 : num_clusters);

    for i = 1 : n
        tmp = zeros(1,n);
        for j = 1 : n
            tmp(j) = norm(F(i,:)- F(j,:),2);
        end
        Q(i,:)=tmp;
    end


    %update R
    G = probs.^gamma;
    [Ur, ~, Vr] = svd(F'*G,'econ');
    R = Ur*Vr';

    %% updata P
    E = zeros(n, num_clusters);
    for ei = 1 : n
        for ec = 1 : num_clusters
            E(ei,ec) = norm(T(ec,:) - F(ei,:) * R , 2)^2;
        end
    end

    if gamma == 1
        for yi = 1 : n
            [~, yindex] = min(E(yi,:));
            probs(yi, yindex) = 1;% n*c result
        end
        [~, labels] = max(Probs,[],2);
    else
        Yup = E.^(1/(1-gamma)); % n * k
        Ydown = sum(Yup, 2);% n * 1 //sum of a row
        probs = Yup ./ repmat(Ydown,1, num_clusters); % n * k result
        
        [~, labels] = max(probs, [], 2);% 
    end

                    
     % update Lagrange multiplier
    Y = Y + mu * (J - Z);
    
    % update penalty parameter
    mu = min(rho * mu, max_mu);
      
%     if(iter > 1)
%         diff_value = abs(last_ratio - ratio);
%         if (diff_value < 1e-6)
% %             disp(diff_value);
%         end
%     end
%     last_ratio = ratio;
    
    err = max(max(abs(J - Z)));
    iter = iter + 1; 
    if err < tol
        break;
    end  
    
    z_con_values(iter) = err;
    obj = length(find(abs(Z) > 1e-6)) + lambda *  norm((X -D * Z), 'fro') + beta * norm(D, 'fro');
    err = abs(obj - obj_tmp) / abs(obj);
%     disp(err);
    obj_values(iter) = err;
    obj_tmp = obj;   
   
end

end


