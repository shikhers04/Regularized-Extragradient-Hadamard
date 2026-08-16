%% Boxplot Experiments for TQY, REMB and REMD
%
% For each method:
%   - 10 initial lambda values
%   - 30 random initial points for each lambda
%   - iteration count and computational time are recorded
%
% Stopping criterion for all methods:
%
%       Er(k) = d(x_{k+1},x_k) <= epsilon
%
% Six separate figures are generated:
%   1. TQY  - iterations
%   2. TQY  - time
%   3. REMB - iterations
%   4. REMB - time
%   5. REMD - iterations
%   6. REMD - time
%
% Author: Shikher Sharma
% -------------------------------------------------------------------------

clc;
clear;
close all;
warning('off');


%% ========================================================================
%  Problem Parameters
%  ========================================================================

N       = 100;      % Dimension
epsilon = 1e-8;     % Stopping tolerance
max_itr = 1000;     % Maximum number of iterations

delta = 0.1;        % TQY parameter


%% ========================================================================
%  Lambda Values and Experiment Settings
%  ========================================================================

lambda_vals = 3 * linspace(0.04, 0.1, 10);

% Alternatively:
% lambda_vals = (1/2).^(1:10);

num_lambdas = length(lambda_vals);
num_trials  = 30;


%% ========================================================================
%  Storage
%  ========================================================================

% TQY
iteration_counts_TQY = zeros(num_trials, num_lambdas);
time_counts_TQY      = zeros(num_trials, num_lambdas);

% REMB
iteration_counts_REMB = zeros(num_trials, num_lambdas);
time_counts_REMB      = zeros(num_trials, num_lambdas);

% REMD
iteration_counts_REMD = zeros(num_trials, num_lambdas);
time_counts_REMD      = zeros(num_trials, num_lambdas);


%% ========================================================================
%  Main Experiment Loop
%  ========================================================================

for i = 1:num_lambdas

    fprintf('\n============================================================\n');
    fprintf('Lambda %d/%d = %.6f\n', ...
        i, num_lambdas, lambda_vals(i));
    fprintf('============================================================\n');

    for j = 1:num_trials

        fprintf('Trial %d/%d\n', j, num_trials);

        % ---------------------------------------------------------------
        % Same random initial point for all three algorithms
        % ---------------------------------------------------------------

        x0 = randi([5, 20], N, 1);


        %% ===============================================================
        %  Algorithm 1: REMB
        %  ===============================================================

        lambda_REMB = lambda_vals(i);
        x_REMB      = x0;

        tic;

        for k = 1:max_itr

            % Store previous iterate
            x_prev_REMB = x_REMB;

            % First step
            y_REMB = J(x_REMB, lambda_REMB);

            % Second step
            x_REMB = JF(y_REMB, x_REMB, lambda_REMB);

            % Error between consecutive iterates
            err_REMB = d(x_prev_REMB, x_REMB);

            % Stopping criterion
            if err_REMB <= epsilon
                break;
            end

        end

        time_counts_REMB(j, i)      = toc;
        iteration_counts_REMB(j, i) = k;


        %% ===============================================================
        %  Algorithm 2: REMD
        %  ===============================================================

        lambda_REMD = lambda_vals(i);
        x_REMD      = x0;

        tic;

        for k = 1:max_itr

            % Store previous iterate
            x_prev_REMD = x_REMD;

            % First step
            y_REMD = J(x_REMD, lambda_REMD / 2);

            % Second step
            x_REMD = JF(y_REMD, x_REMD, lambda_REMD);

            % Error between consecutive iterates
            err_REMD = d(x_prev_REMD, x_REMD);

            % Stopping criterion
            if err_REMD <= epsilon
                break;
            end

        end

        time_counts_REMD(j, i)      = toc;
        iteration_counts_REMD(j, i) = k;


        %% ===============================================================
        %  Algorithm 3: TQY
        %  ===============================================================

        % Reset adaptive stepsize for every new trial
        lambda_TQY = lambda_vals(i);

        x_TQY = x0;

        tic;

        for k = 1:max_itr

            % Store previous iterate
            x_prev_TQY = x_TQY;

            % Adaptive parameters
            xi_k    = 1 + 1 / k^3;
            sigma_k = 1 / (k + 1000)^2;

            % Extragradient steps
            y_TQY = JF(x_TQY, x_TQY, lambda_TQY);

            x_TQY = JF(y_TQY, x_TQY, lambda_TQY);

            % Quantity used in adaptive stepsize rule
            D = F(x_prev_TQY, x_TQY) ...
              - F(x_prev_TQY, y_TQY) ...
              - F(y_TQY, x_TQY);

            % Adaptive stepsize
            if D > 0

                L_1 = delta ...
                    * d(x_prev_TQY, y_TQY) ...
                    * d(x_TQY, y_TQY) / D;

                L_2 = xi_k * lambda_TQY + sigma_k;

                lambda_TQY = min(L_1, L_2);

            else

                lambda_TQY = ...
                    xi_k * lambda_TQY + sigma_k;

            end

            % Error between consecutive iterates
            err_TQY = d(x_prev_TQY, x_TQY);

            % Stopping criterion
            if err_TQY <= epsilon
                break;
            end

        end

        time_counts_TQY(j, i)      = toc;
        iteration_counts_TQY(j, i) = k;

    end

end


%% ========================================================================
%  Lambda Labels
%  ========================================================================

lambda_labels = arrayfun(@(x) sprintf('%.2f', x), ...
    lambda_vals, 'UniformOutput', false);


%% ========================================================================
%  Figure 1: TQY Iterations
%  ========================================================================

figure;

boxplot(iteration_counts_TQY, ...
    'Labels', lambda_labels);

xlabel('$\lambda_0$', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

ylabel('Number of iterations', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

title('TQY', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

set(gca, 'FontSize', 12);

box on;
grid off;

print(gcf, 'TQY_iterations.eps', '-depsc');


%% ========================================================================
%  Figure 2: TQY Time
%  ========================================================================

figure;

boxplot(time_counts_TQY, ...
    'Labels', lambda_labels);

xlabel('$\lambda_0$', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

ylabel('CPU time (seconds)', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

title('TQY', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

set(gca, 'FontSize', 12);

box on;
grid off;

print(gcf, 'TQY_time.eps', '-depsc');


%% ========================================================================
%  Figure 3: REMB Iterations
%  ========================================================================

figure;

boxplot(iteration_counts_REMB, ...
    'Labels', lambda_labels);

xlabel('$\lambda$', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

ylabel('Number of iterations', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

title('REMB', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

set(gca, 'FontSize', 12);

box on;
grid off;

print(gcf, 'REMB_iterations.eps', '-depsc');


%% ========================================================================
%  Figure 4: REMB Time
%  ========================================================================

figure;

boxplot(time_counts_REMB, ...
    'Labels', lambda_labels);

xlabel('$\lambda$', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

ylabel('CPU time (seconds)', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

title('REMB', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

set(gca, 'FontSize', 12);

box on;
grid off;

print(gcf, 'REMB_time.eps', '-depsc');


%% ========================================================================
%  Figure 5: REMD Iterations
%  ========================================================================

figure;

boxplot(iteration_counts_REMD, ...
    'Labels', lambda_labels);

xlabel('$\lambda$', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

ylabel('Number of iterations', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

title('REMD', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

set(gca, 'FontSize', 12);

box on;
grid off;

print(gcf, 'REMD_iterations.eps', '-depsc');


%% ========================================================================
%  Figure 6: REMD Time
%  ========================================================================

figure;

boxplot(time_counts_REMD, ...
    'Labels', lambda_labels);

xlabel('$\lambda$', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

ylabel('CPU time (seconds)', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

title('REMD', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

set(gca, 'FontSize', 12);

box on;
grid off;

print(gcf, 'REMD_time.eps', '-depsc');


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
%D Logarithmic distance between x and y.
%
%   d(x,y) = sqrt(sum_i [log(x_i/y_i)]^2)

    value = sqrt(sum((log(x ./ y)).^2));

end