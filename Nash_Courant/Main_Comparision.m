clc;
clear;
close all;
warning('off');

%% ============================================================
% PARAMETERS@2
% =============================================================

N  = 4;
%x0=round(500*rand(4,1))+500;
 x0 = [570; 948; 503; 812];
% x0 = [620; 932; 511; 808]; 
% x0 = [558; 786; 641; 956]; 
% x0 = [875; 859; 959; 816];

max_inner = 10;
max_itr   = 1000;

tol_Prop = 1e-8;
tol_Neto = 1e-2;
tol_TQY  = 1e-8;

%% Busemann and Distance parameters

lambda = @(k) 0.8+ 1 / (k + 1000)^2;
eta    = @(k) 1 / k; % ^(1/2);

%% TQY parameters

tau   = 0.01;
delta = 0.1;
chi   = 1.5;

xi    = @(k) 1 + 1 / k^3;
sigma = @(k) 1 / (k + 1000)^2;

%% ============================================================
% FEASIBLE SET
% =============================================================

% lowerBound = 1e-8;
% 
% lb = lowerBound * ones(N,1);
% ub = inf(N,1);
% 
% A      = -eye(N);
% b_cons = -lowerBound * ones(N,1);


lowerBound = 1e-8;

lb = [1000;500;800;500];
ub = [2000;2500;1500;3000];

% A      = -eye(N);
% b_cons = -lowerBound * ones(N,1);
%% ============================================================
% FMINCON OPTIONS
% =============================================================

options = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','off', ...
    'OptimalityTolerance',1e-10, ...
    'StepTolerance',1e-12, ...
    'ConstraintTolerance',1e-10, ...
    'MaxIterations',1000, ...
    'MaxFunctionEvaluations',10000);

%% ============================================================
% METHOD 1: BUSEMANN REGULARIZATION + NETO
% =============================================================

fprintf('\n');
fprintf('=============================================\n');
fprintf(' METHOD 1: BUSEMANN + NETO\n');
fprintf('=============================================\n');

x_b = x0;
errors_b = zeros(max_itr,1);

tic;

for k = 1:max_itr

    x_prev_b = x_b;

    lambda_n = lambda(k);
    eta_n    = eta;

    F_b = @(x,y) ...
        lambda_n * F(x,y) ...
        + d(x,x_prev_b) * Busemann(y,x,x_prev_b);

    [y_b, K_Busemann, Err_Busemann] = ...
        Neto_extragradient_step( ...
        F_b, @d, lb, ub, x_b, ...
        eta_n, max_inner, tol_Neto);

    objective_b = @(q) ...
        F(y_b,q) ...
        + (1 / (2 * lambda_n)) * d(x_prev_b,q)^2;

    q0 = max(y_b,lb);

    [x_next,~,exitflag_b] = fmincon( ...
        objective_b, q0, ...
        [],[],[],[],lb,ub,[],options);

    if exitflag_b <= 0
        warning(['Busemann iteration %d: computation of ', ...
            'x_{n+1} did not converge.'],k);
    end

    x_b = x_next(:);

    err_b = d(x_prev_b,x_b);
    errors_b(k) = err_b;

    
    if err_b <= tol_Prop
        break;
    end

end

time_Busemann = toc;

iter_Busemann = k;
errors_b = errors_b(1:iter_Busemann);

x_Busemann = x_b;

fprintf('Busemann + Neto: Iterations = %d, Final Error = %.4e, Time = %.6f sec\n', ...
    iter_Busemann, errors_b(end), time_Busemann);

%% ============================================================
% METHOD 2: DISTANCE REGULARIZATION + NETO
% =============================================================

fprintf('\n');
fprintf('=============================================\n');
fprintf(' METHOD 2: DISTANCE + NETO\n');
fprintf('=============================================\n');

x_d = x0;
errors_d = zeros(max_itr,1);

tic;

for k = 1:max_itr

    x_prev_d = x_d;

    lambda_n = lambda(k);
    eta_n    = eta;

    F_d = @(x,y) ...
        lambda_n * F(x,y) ...
        + d(y,x_prev_d)^2 ...
        - d(x,x_prev_d)^2;

    [y_d, K_Distance, Err_Distance] = ...
        Neto_extragradient_step( ...
        F_d, @d, lb, ub, x_d, ...
        eta_n, max_inner, tol_Neto);

    objective_d = @(q) ...
        F(y_d,q) ...
        + (1 / (2 * lambda_n)) * d(x_prev_d,q)^2;

    q0 = max(y_d,lb);

    [x_next,~,exitflag_d] = fmincon( ...
        objective_d, q0, ...
        [],[],[],[],lb,ub,[],options);

    if exitflag_d <= 0
        warning(['Distance iteration %d: computation of ', ...
            'x_{n+1} did not converge.'],k);
    end

    x_d = x_next(:);

    err_d = d(x_prev_d,x_d);
    errors_d(k) = err_d;

    if err_d <= tol_Prop
        break;
    end

end

time_Distance = toc;

iter_Distance = k;
errors_d = errors_d(1:iter_Distance);

x_Distance = x_d;

fprintf('Diatance + Neto: Iterations = %d, Final Error = %.4e, Time = %.6f sec\n', ...
    iter_Distance, errors_d(end), time_Distance);

%% ============================================================
% METHOD 3: TQY / YAO EXTRAGRADIENT
% =============================================================

fprintf('\n');
fprintf('=============================================\n');
fprintf(' METHOD 3: TQY / YAO\n');
fprintf('=============================================\n');

F_y = @(x,y) F(x,y);

tic;

[x_TQY, iter_TQY, errors_TQY] = ...
    TQY_extragradient( ...
    F_y, @d, lb, ub, x0, ...
    tau, delta, chi, xi, sigma, ...
    max_itr, tol_TQY);

time_TQY = toc;

x_TQY = x_TQY(:);

fprintf(['TQY / Yao: iterations = %d, ', ...
         'final error = %.4e, ', ...
         'Time = %.6f sec\n'], ...
         iter_TQY, errors_TQY(end), time_TQY);

%% ============================================================
% COMPARISON
% =============================================================

fprintf('\n');
fprintf('==========================================================================\n');
fprintf('                             COMPARISON\n');
fprintf('==========================================================================\n');

fprintf('%-25s %-15s %-18s %-15s\n', ...
    'Method','Iterations','Final Error','Time (sec)');

fprintf('--------------------------------------------------------------------------\n');

fprintf('%-25s %-15d %-18.6e %-15.6f\n', ...
    'Busemann + Neto', ...
    iter_Busemann, ...
    errors_b(end), ...
    time_Busemann);

fprintf('%-25s %-15d %-18.6e %-15.6f\n', ...
    'Distance + Neto', ...
    iter_Distance, ...
    errors_d(end), ...
    time_Distance);

fprintf('%-25s %-15d %-18.6e %-15.6f\n', ...
    'TQY / Yao', ...
    iter_TQY, ...
    errors_TQY(end), ...
    time_TQY);

fprintf('==========================================================================\n');

%% ============================================================
% FINAL ITERATES
% =============================================================

fprintf('\nFinal iterate: Busemann + Neto\n');
disp(x_Busemann);

fprintf('Final iterate: Distance + Neto\n');
disp(x_Distance);

fprintf('Final iterate: TQY / Yao\n');
disp(x_TQY);



%% ============================================================
% CONVERGENCE PLOT
% =============================================================

figure;

semilogy( ...
    1:iter_Busemann, ...
    errors_b, ...
    'LineWidth',1.5, ...
    'MarkerSize',5);

hold on;

semilogy( ...
    1:iter_Distance, ...
    errors_d, ...
    'LineWidth',1.5, ...
    'MarkerSize',5);

semilogy( ...
    1:iter_TQY, ...
    errors_TQY, ...
    'LineWidth',1.5, ...
    'MarkerSize',5);

xlabel('Iteration (n)');
ylabel('Er(n)');

%title('Convergence histories');

legend( ...
    'Alg. REMB', ...
    'Alg. REMD', ...
    'Alg. TQY', ...
    'Location','best');

grid off;
box on;

hold off;

%% ============================================================
% LOCAL FUNCTIONS
% =============================================================

% function f = F(x,y)
% 
%     if any(x <= 0) || any(y <= 0)
%         f = 1e20;
%         return;
%     end
% 
%     f = sum(log(x) .* log(y ./ x));
% 
% end

function f = F(x, y)

P = [0.01   0.01   0.01   0.01;
    0.02   0.02   0.02   0.02;
    0.015  0.015  0.015  0.015;
    0.05   0.05   0.05   0.05];

Q = [0.01   0      0      0;
    0      0.02   0      0;
    0      0      0.015  0;
    0      0      0      0.05];

r = [-80;
    -95;
    -83;
    -95];


% % Matrices
% P = [1 0;0 2];
% 
% Q = [3 0;0 5];
% 
% % P = [4 1;1 5];
% % 
% % Q = [2 1;1 2];
% 
% % Vector r
% r = [-1;-2];

% Bifunction
f = (P*x + Q*y + r)' * (y - x);

end


function value = Busemann(y,z,x)

    if any(x <= 0) || any(y <= 0) || any(z <= 0)
        value = 1e20;
        return;
    end

    direction = log(x ./ z);

    denominator = sqrt(sum(direction.^2));

    if denominator < 1e-14
        value = 0;
        return;
    end

    value = ...
        sum(direction .* log(z ./ y)) ...
        / denominator;

end


function D = d(x,y)

    if any(x <= 0) || any(y <= 0)
        D = 1e20;
        return;
    end

    D = sqrt(sum((log(x ./ y)).^2));

end