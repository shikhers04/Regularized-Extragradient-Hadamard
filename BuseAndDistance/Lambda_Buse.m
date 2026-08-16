clc;
clear;
warning('off');
format long

N = 3;
%x0 = 5 * ones(N, 1);
%x0=[5;5;5];
epsilon = 1e-8;
max_itr = 1000;
num_trials = 30;
lambda_vals = 3 * linspace(0.01, 0.1, 10);
num_lambdas = length(lambda_vals);

% Preallocate
iteration_counts = zeros(num_trials, num_lambdas);
comp_times = zeros(num_trials, num_lambdas);
error_curves = cell(num_lambdas, 1);

for i = 1:num_lambdas
    lambda = lambda_vals(i);

    for j = 1:num_trials
        x_s = randi([5, 20], N, 1);
        %x_s=[5;5;5];
        ww_s = zeros(max_itr, 1);
        tic;
        for k = 1:max_itr
            x_prev_s = x_s;
            y_s = J(x_s, lambda);
            x_s = JF(y_s, x_s, lambda);

            err_s = d(x_prev_s, x_s);
            ww_s(k) = err_s;

            if err_s <= epsilon
                ww_s = ww_s(1:k);
                break;
            end
        end
        comp_times(j, i) = toc;
        iteration_counts(j, i) = k;

        if j == 1
            error_curves{i} = ww_s;
        end
    end
end

% Compute mean and std for iterations and computation time
mean_iters = mean(iteration_counts);
std_iters = std(iteration_counts);
mean_times = mean(comp_times);
std_times = std(comp_times);

% Print LaTeX table rows
fprintf('lambda & Mean Iterations & Std Dev & Mean Time (s) & Std Dev \\\\ \\hline\n');
for i = 1:num_lambdas
    fprintf('%.2f & %.f & %.2f & %.6f & %.6f \\\\ \n', ...
        lambda_vals(i), mean_iters(i), std_iters(i), mean_times(i), std_times(i));
end


% %% === Plot 1: Convergence Curves for All Lambda Values ===
% figure;
% hold on;
% colors = lines(num_lambdas);
% for i = 1:num_lambdas
%     curve = error_curves{i};
%     semilogy(1:length(curve), curve, 'LineWidth', 2, 'Color', colors(i, :));
% end
% set(gca, 'YScale', 'log')
% xlabel('Iteration (n)', 'FontSize', 14);
% ylabel('Er(n)', 'FontSize', 14);
% title('Convergence of SGR Iteration for Different \lambda', 'FontSize', 14);
% legend(cellstr(num2str(lambda_vals', '\\lambda = %.3f')), 'Location', 'northeastoutside','FontSize', 14);
% box on;
% grid off;
% hold off;

% %%% === Plot 2: Boxplot for Iteration Counts ===
% figure;
% boxplot(iteration_counts, ...
%     'Labels', cellstr(num2str(lambda_vals', '%.3f')), ...
%     'Symbol', '+', ...
%     'Widths', 0.5);
% xlabel('\lambda', 'FontSize', 14);
% ylabel('Iterations', 'FontSize', 14);
% %title('Boxplot of Iterations to Convergence (SGR)', 'FontSize', 16);
% grid on;
% box on;
% 
% % %% === Plot 3: Boxplot for Computational Time ===
% figure;
% boxplot(comp_times, ...
%     'Labels', cellstr(num2str(lambda_vals', '%.3f')), ...
%     'Symbol', '+', ...
%     'Widths', 0.5);
% xlabel('\lambda', 'FontSize', 14);
% ylabel('Time (seconds)', 'FontSize', 14);
% %title('Boxplot of Computational Time (SGR)', 'FontSize', 16);
% grid on;
% box on;


%%% === Auxiliary Functions ===
function j = J(x, t)
    p = 1 / (1 + 3 * t);
    j1 = (x(1) * x(2)^(3*t) * x(3)^(3*t))^p;
    j2 = (x(1)^(3*t) * x(2)^(-1) * x(3)^(3*t))^p;
    j3 = (x(1)^(3*t) * x(2)^(3*t) * x(3)^(1 + 6*t))^p;
    j = [j1; j2; j3];
end

function jf = JF(x, y, t)
    factor = (x(1) * x(2) / x(3))^(3 * t);
    prox1 = y(1) / factor;
    prox2 = y(2) / factor;
    prox3 = y(3) * factor;
    jf = [prox1; prox2; prox3];
end

function D = d(x, y)
    D = sum((log(x ./ y)).^2);
end




% % Monotone Operator F
% function f = F(x, y)
%     % x, y: vectors of length 3
%     l = log((x(1)*x(2)) / x(3));
%     f = [
%         3 * l * log(y(1) / x(1));
%         3 * l * log(y(2) / x(2));
%         -3 * l * log(y(3) / x(3))
%     ];
% end
% 
% 
% function D = d(x, y)
%     D = sum((log(x ./ y)).^2);
% end