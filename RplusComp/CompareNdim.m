%% Comparison of TQY, REMB, and REMD Algorithms
%
% This script compares the numerical performance of three algorithms:
%
%   1. TQY  - Adaptive extragradient-type method
%   2. REMB - Regularized Extragradient Method (Busemann version)
%   3. REMD - Regularized Extragradient Method (distance version)
%
% The algorithms are tested on an equilibrium problem defined on the
% positive orthant equipped with the logarithmic metric.
%
% Author: Shikher Sharma
% -------------------------------------------------------------------------

clc;
clear;
close all;
warning('off');

%% Problem Parameters

N       = 100;              % Dimension
x0      = randi([5, 20], N, 1);   %5 * ones(N, 1);   % Initial point
p       = ones(N, 1);       % Reference solution
epsilon = 1e-8;             % Stopping tolerance
max_itr = 1000;             % Maximum number of iterations

%% Algorithm Parameters

lambda = 0.18;
delta  = 0.1;

%% ========================================================================
%  Algorithm 1: REMB
%  ========================================================================

lambda_REMB = lambda;
x_REMB      = x0;

error_REMB = zeros(max_itr, 1);

tic;

for k = 1:max_itr

    % First step
    y_REMB = J(x_REMB, lambda_REMB);

    % Second step
    x_REMB = JF(y_REMB, x_REMB, lambda_REMB);

    % Error
    error_REMB(k) = d(p, x_REMB);

    % Stopping criterion
    if error_REMB(k) <= epsilon
        error_REMB = error_REMB(1:k);
        break;
    end

end

time_REMB = toc;
iter_REMB = k;


%% ========================================================================
%  Algorithm 2: REMD
%  ========================================================================

lambda_REMD = lambda;
x_REMD      = x0;

error_REMD = zeros(max_itr, 1);

tic;

for k = 1:max_itr

    % First step
    y_REMD = J(x_REMD, lambda_REMD / 2);

    % Second step
    x_REMD = JF(y_REMD, x_REMD, lambda_REMD);

    % Error
    error_REMD(k) = d(p, x_REMD);

    % Stopping criterion
    if error_REMD(k) <= epsilon
        error_REMD = error_REMD(1:k);
        break;
    end

end

time_REMD = toc;
iter_REMD = k;


%% ========================================================================
%  Algorithm 3: TQY
%  ========================================================================

lambda_TQY = lambda;
x_TQY      = x0;

error_TQY = zeros(max_itr, 1);

tic;

for k = 1:max_itr

    x_prev = x_TQY;

    % Adaptive parameters
    xi_k    = 1 + 1 / k^3;
    sigma_k = 1 / (k + 1000)^2;

    % Extragradient steps
    y_TQY = JF(x_TQY, x_TQY, lambda_TQY);
    x_TQY = JF(y_TQY, x_TQY, lambda_TQY);

    % Quantity used in the adaptive stepsize rule
    D = F(x_prev, x_TQY) ...
        - F(x_prev, y_TQY) ...
        - F(y_TQY, x_TQY);

    % Adaptive stepsize
    if D > 0

        L_1 = delta * d(x_prev, y_TQY) * d(x_TQY, y_TQY) / D;
            

        L_2 = xi_k * lambda_TQY + sigma_k;
            

        lambda_TQY = min(L_1, L_2);

    else

        lambda_TQY = xi_k * lambda_TQY + sigma_k;

    end
    

    % Error
    error_TQY(k) = d(p, x_TQY);

    % Stopping criterion
    if error_TQY(k) <= epsilon
        error_TQY = error_TQY(1:k);
        break;
    end

end

time_TQY = toc;
iter_TQY = k;


%% ========================================================================
%  Numerical Results
%  ========================================================================

fprintf('\n');
fprintf('==============================================================\n');
fprintf('                     NUMERICAL RESULTS\n');
fprintf('==============================================================\n');
fprintf('%-12s %-15s %-18s %-15s\n', ...
        'Method', 'Iterations', 'Final Error', 'Time (sec)');
fprintf('--------------------------------------------------------------\n');

fprintf('%-12s %-15d %-18.6e %-15.6f\n', ...
        'TQY', iter_TQY, error_TQY(end), time_TQY);

fprintf('%-12s %-15d %-18.6e %-15.6f\n', ...
        'REMB', iter_REMB, error_REMB(end), time_REMB);

fprintf('%-12s %-15d %-18.6e %-15.6f\n', ...
        'REMD', iter_REMD, error_REMD(end), time_REMD);

fprintf('==============================================================\n');


%% ========================================================================
%  Convergence Plot
%  ========================================================================

figure;

semilogy(1:length(error_TQY), ...
         error_TQY, ...
         'LineWidth', 2);

hold on;

semilogy(1:length(error_REMB), ...
         error_REMB, ...
         'LineWidth', 2);

semilogy(1:length(error_REMD), ...
         error_REMD, ...
         'LineWidth', 2);

hold off;

xlabel('Iteration ($n$)', ...
       'Interpreter', 'latex', ...
       'FontSize', 14);

ylabel('$\mathrm{Er}(n)$', ...
       'Interpreter', 'latex', ...
       'FontSize', 14);

legend({'TQY', 'REMB', 'REMD'}, ...
       'Interpreter', 'latex', ...
       'Location', 'best');

grid off;
box on;

set(gca, 'FontSize', 12);
print(gcf, 'figure.eps', '-depsc');


%% ========================================================================
%  Auxiliary Functions
%  ========================================================================

function j = J(x, t)
%J Resolvent used in the REMB/REMD iteration.
%
%   J(x,t) = x^(1/(1+t))

    j = x .^ (1 / (1 + t));

end


function jf = JF(x, y, t)
%JF Proximal/resolvent-type mapping.
%
%   JF(x,y,t) = y .* x^(-t)

    jf = y .* x .^ (-t);

end


function value = F(x, y)
%F Bifunction defining the equilibrium problem.
%
%   F(x,y) = sum_i log(x_i) log(y_i/x_i)

    value = sum(log(x) .* log(y ./ x));

end


function value = d(x, y)
%D  logarithmic distance between x and y.
%
%   d(x,y) = sqrt(sum_i [log(x_i/y_i)]^2)

    value = sqrt(sum((log(x ./ y)).^2));

end