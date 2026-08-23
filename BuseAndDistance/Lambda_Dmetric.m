%% Example 1: Lambda (regularization parameter) study with distance-metric regularization
% This script implements the numerical experiment corresponding to
% Example 1 of the associated paper.
%
% The experiment evaluates the number of iterations and computational time
% for a range of regularization parameters lambda. The convergence error is
% measured by the squared logarithmic distance
%
%     d(x,y) = sqrt(sum((log(x_i/y_i)).^2)).
%
% Three plotting sections are retained below as commented code:
%   Plot 1 - convergence curves,
%   Plot 2 - boxplot of iteration counts, and
%   Plot 3 - boxplot of computational times.
%
% Requirements:
%   MATLAB R2020b or later (local functions in scripts are used).

clc;
clear;
close all;
format shorte;

%% Parameters
N = 3;
epsilon = 1e-8;
max_itr = 1000;
num_trials = 30;
lambda_vals = 3 * linspace(0.01, 0.10, 10);
num_lambdas = numel(lambda_vals);

% Set the random seed for reproducibility.
RNG_SEED = 1;
rng(RNG_SEED, 'twister');

%% Preallocation
iteration_counts = zeros(num_trials, num_lambdas);
comp_times = zeros(num_trials, num_lambdas);
error_curves = cell(num_lambdas, 1);

%% Numerical experiment
for i = 1:num_lambdas
    lambda = lambda_vals(i);

    for j = 1:num_trials
        % Random initial point in the positive orthant.
        x_s = randi([5, 20], N, 1);
        % x_s = randi([100, 500], N, 1);
        % x_s = [1; 2; 3];

        error_history = zeros(max_itr, 1);

        tic;
        for k = 1:max_itr
            x_prev_s = x_s;

            % Distance-metric regularized step.
            y_s = J(x_s, lambda / 2);
            x_s = JF(y_s, x_s, lambda);

            % Stopping criterion: d(x_{k-1},x_k) <= epsilon.
            err_s = d(x_prev_s, x_s);
            error_history(k) = err_s;

            if err_s <= epsilon
                error_history = error_history(1:k);
                break;
            end
        end
        comp_times(j, i) = toc;
        iteration_counts(j, i) = k;

        % Store one representative convergence curve for each lambda.
        if j == 1
            error_curves{i} = error_history;
        end
    end
end

%% Summary statistics
mean_iters = mean(iteration_counts, 1);
std_iters = std(iteration_counts, 0, 1);
mean_times = mean(comp_times, 1);
std_times = std(comp_times, 0, 1);

%% Display results in LaTeX-table format
fprintf('\nLaTeX table rows:\n');
fprintf('lambda & Mean Iterations & Std Dev & Mean Time (s) & Std Dev \\\\ \\hline\n');

for i = 1:num_lambdas
    fprintf('%.2f & %.0f & %.2f & %.6f & %.6f \\\\ \n', ...
        lambda_vals(i), mean_iters(i), std_iters(i), ...
        mean_times(i), std_times(i));
end

%% === Plot 1: Convergence Curves for All Lambda Values ===
% Uncomment this section to plot the convergence histories.
% figure;
% hold on;
% colors = lines(num_lambdas);
% for i = 1:num_lambdas
%     curve = error_curves{i};
%     semilogy(1:numel(curve), curve, 'LineWidth', 2, ...
%         'Color', colors(i, :));
% end
% xlabel('Iteration (n)', 'FontSize', 14);
% ylabel('Er(n)', 'FontSize', 14);
% title('Convergence for Different \lambda', 'FontSize', 14);
% legend(cellstr(num2str(lambda_vals', '\\lambda = %.3f')), ...
%     'Location', 'northeastoutside', 'FontSize', 12);
% box on;
% grid off;
% hold off;

%% === Plot 2: Boxplot for Iteration Counts ===
% Uncomment this section to compare iteration counts across lambda values.
% figure;
% boxplot(iteration_counts, ...
%     'Labels', cellstr(num2str(lambda_vals', '%.3f')), ...
%     'Symbol', '+', ...
%     'Widths', 0.5);
% xlabel('\lambda', 'FontSize', 14);
% ylabel('Iterations', 'FontSize', 14);
% grid on;
% box on;

%% === Plot 3: Boxplot for Computational Time ===
% Uncomment this section to compare computational times across lambda values.
% figure;
% boxplot(comp_times, ...
%     'Labels', cellstr(num2str(lambda_vals', '%.3f')), ...
%     'Symbol', '+', ...
%     'Widths', 0.5);
% xlabel('\lambda', 'FontSize', 14);
% ylabel('Time (seconds)', 'FontSize', 14);
% grid on;
% box on;

%% Local functions
function j = J(x, t)
    % Resolvent-type mapping used in Example 1.
    p = 1 / (1 + 3 * t);

    j1 = (x(1) * x(2)^(3 * t) * x(3)^(3 * t))^p;
    j2 = (x(1)^(3 * t) * x(2)^(-1) * x(3)^(3 * t))^p;
    j3 = (x(1)^(3 * t) * x(2)^(3 * t) * x(3)^(1 + 6 * t))^p;

    j = [j1; j2; j3];
end

function jf = JF(x, y, t)
    % Distance-metric regularized correction step.
    factor = (x(1) * x(2) / x(3))^(3 * t);

    prox1 = y(1) / factor;
    prox2 = y(2) / factor;
    prox3 = y(3) * factor;

    jf = [prox1; prox2; prox3];
end

function D = d(x, y)
    % Squared logarithmic distance on the positive orthant.
    D = sum((log(x ./ y)).^2);
end
