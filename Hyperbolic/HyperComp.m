%% Comparison of REMB, REMD, and TQY on Hyperbolic Space H^N
%
% This script compares the numerical performance of three algorithms:
%
%   1. REMB - Regularized Extragradient Method with Busemann regularization
%   2. REMD - Regularized Extragradient Method with distance regularization
%   3. TQY  - Adaptive extragradient-type method
%
% The numerical experiment is carried out on the hyperboloid model
% of the N-dimensional hyperbolic space
%
%       H^N = {x in R^(N+1) : <x,x>_L = -1, x_{N+1} > 0},
%
% where the Lorentz inner product is
%
%       <x,y>_L = sum_{i=1}^N x_i y_i - x_{N+1} y_{N+1}.
%
% The stopping criterion for all algorithms is
%
%       d_H(x_{k+1},x_k) <= tolerance.
%
% MATLAB requirement:
%   Local functions in scripts require MATLAB R2016b or later.
%
% -------------------------------------------------------------------------

clc;
clearvars;
close all;
format shorte;


%% ========================================================================
%  PARAMETERS
% =========================================================================

tolerance = 1e-8;       % Stopping tolerance
max_itr   = 300;        % Maximum number of iterations
N         = 50;         % Dimension of H^N


%% ========================================================================
%  INITIAL POINT AND REFERENCE POINT
% =========================================================================

% Random initial point x0 in H^N
%
% A point x = (x_1,...,x_N,x_{N+1}) belongs to H^N if
%
%       sum_{i=1}^N x_i^2 - x_{N+1}^2 = -1
%
% with x_{N+1} > 0.

w = randn(N,1);

x_last = sqrt(1 + sum(w.^2));

x0 = [w; x_last];


% Reference point a_ref in H^N
a_ref = zeros(N+1,1);
a_ref(end) = 1;

% w = randn(N,1);
% x_last = sqrt(1 + sum(w.^2));
% a_ref = [w;x_last];



%% ========================================================================
%  ALGORITHM 1: REMB
% =========================================================================

x_REMB = x0;

errors_REMB = zeros(max_itr,1);

tic;

for k = 1:max_itr

    % Regularization parameter
    lambda_k = 1.1 + 1 / (k + 1000)^2;

    % Store previous iterate
    x_prev = x_REMB;

    % First step: Busemann-type resolvent
    y_REMB = J_hyperbolic(x_REMB, a_ref, lambda_k);

    % Second step: proximal mapping
    x_REMB = JF(y_REMB, a_ref, lambda_k);

    % Error between two consecutive iterates
    errors_REMB(k) = dH(x_prev, x_REMB);

    % Stopping criterion
    if errors_REMB(k) <= tolerance
        break;
    end

end

time_REMB = toc;

itr_REMB = k;

errors_REMB = errors_REMB(1:itr_REMB);

final_error_REMB = errors_REMB(end);


%% ========================================================================
%  ALGORITHM 2: REMD
% =========================================================================

x_REMD = x0;

errors_REMD = zeros(max_itr,1);

tic;

for k = 1:max_itr

    % Regularization parameter
    lambda_k = 1.1 + 1 / (k + 1000)^2;

    % Store previous iterate
    x_prev = x_REMD;

    % First step: distance-type resolvent
    y_REMD = K(x_REMD, a_ref, lambda_k);

    % Second step: proximal mapping
    x_REMD = JF(y_REMD, a_ref, lambda_k);

    % Error between two consecutive iterates
    errors_REMD(k) = dH(x_prev, x_REMD);

    % Stopping criterion
    if errors_REMD(k) <= tolerance
        break;
    end

end

time_REMD = toc;

itr_REMD = k;

errors_REMD = errors_REMD(1:itr_REMD);

final_error_REMD = errors_REMD(end);


%% ========================================================================
%  ALGORITHM 3: TQY
% =========================================================================

x_TQY = x0;

lambda_TQY = 0.1;

delta = 0.1;
chi   = 1.5;

errors_TQY = zeros(max_itr,1);

tic;

for k = 1:max_itr

    % Store previous iterate
    x_prev = x_TQY;

    % Adaptive parameters
    xi_k    = 1 + 1 / k^2;
    sigma_k = 1 / (k + 1000)^2;

    % First step
    y_TQY = JF(x_TQY, a_ref, lambda_TQY);

    % Second step
    x_TQY = JF(y_TQY, a_ref, chi * lambda_TQY);

    % --------------------------------------------------------------------
    % Adaptive quantity
    %
    % D_k = F(x_k,x_{k+1})
    %       - F(x_k,y_k)
    %       - F(y_k,x_{k+1}).
    %
    % For
    %
    %       F(x,y) = 1/2 [d_H(y,a)^2 - d_H(x,a)^2],
    %
    % these terms telescope, so D_k is theoretically zero.
    %
    % We nevertheless compute D_k explicitly so that this code can
    % readily be used with other bifunctions.
    % --------------------------------------------------------------------

    D_k = F(x_prev, x_TQY, a_ref) ...
        - F(x_prev, y_TQY, a_ref) ...
        - F(y_TQY, x_TQY, a_ref);

    % Numerical tolerance for deciding whether D_k is positive
    D_tolerance = 1e-14;

    if D_k > D_tolerance

        candidate_1 = ...
            delta * dH(x_prev, y_TQY) ...
            * dH(x_TQY, y_TQY) / D_k;

        candidate_2 = ...
            xi_k * lambda_TQY + sigma_k;

        lambda_TQY = min(candidate_1, candidate_2);

    else

        lambda_TQY = xi_k * lambda_TQY + sigma_k;

    end

    % Error between two consecutive iterates
    errors_TQY(k) = dH(x_prev, x_TQY);

    % Stopping criterion
    if errors_TQY(k) <= tolerance
        break;
    end

end

time_TQY = toc;

itr_TQY = k;

errors_TQY = errors_TQY(1:itr_TQY);

final_error_TQY = errors_TQY(end);


%% ========================================================================
%  DISPLAY FINAL ITERATES
% =========================================================================

fprintf('\n');
fprintf('==========================================================================\n');
fprintf('                         FINAL ITERATES\n');
fprintf('==========================================================================\n');

fprintf('\nREMB:\n');
disp(x_REMB);

fprintf('REMD:\n');
disp(x_REMD);

fprintf('TQY:\n');
disp(x_TQY);


%% ========================================================================
%  DISPLAY NUMERICAL COMPARISON
% =========================================================================

fprintf('==========================================================================\n');
fprintf('                         NUMERICAL COMPARISON\n');
fprintf('==========================================================================\n');

fprintf('%-15s %-15s %-20s %-15s\n', ...
    'Algorithm', 'Iterations', 'Final Error', 'Time (sec)');

fprintf('--------------------------------------------------------------------------\n');

fprintf('%-15s %-15d %-20.6e %-15.6f\n', ...
    'REMB', itr_REMB, final_error_REMB, time_REMB);

fprintf('%-15s %-15d %-20.6e %-15.6f\n', ...
    'REMD', itr_REMD, final_error_REMD, time_REMD);

fprintf('%-15s %-15d %-20.6e %-15.6f\n', ...
    'TQY', itr_TQY, final_error_TQY, time_TQY);

fprintf('==========================================================================\n');


%% ========================================================================
%  CONVERGENCE PLOT
% =========================================================================

figure;

semilogy(1:itr_REMB, errors_REMB, ...
    'LineWidth', 2);

hold on;

semilogy(1:itr_REMD, errors_REMD, ...
    'LineWidth', 2);

semilogy(1:itr_TQY, errors_TQY, ...
    'LineWidth', 2);

xlabel('Iterations', ...
    'FontSize', 14);

ylabel('$d_H(x_{k+1},x_k)$', ...
    'Interpreter', 'latex', ...
    'FontSize', 14);

legend('REMB', 'REMD', 'TQY', ...
    'Location', 'southwest', ...
    'FontSize', 12);

box on;

set(gca, 'FontSize', 12);

hold off;


%% ========================================================================
%  LOCAL FUNCTIONS
% =========================================================================


%% ------------------------------------------------------------------------
%  Busemann-type Hyperbolic Resolvent
% -------------------------------------------------------------------------

function z = J_hyperbolic(x, a, lambda)

    d = dH(x, a);

    % If x and a coincide
    if d < 1e-14
        z = a;
        return;
    end

    alpha = ...
        sinh(d / (1 + lambda)) / sinh(d);

    beta = ...
        sinh((lambda * d) / (1 + lambda)) / sinh(d);

    z = alpha * x + beta * a;

    % Correct possible floating-point drift
    z = projectHN(z);

end


%% ------------------------------------------------------------------------
%  Distance-type Resolvent
% -------------------------------------------------------------------------

function z = K(x, a, lambda)

    d = dH(x, a);

    % If x and a coincide
    if d < 1e-14
        z = a;
        return;
    end

    alpha = ...
        sinh(d / (1 + lambda / 2)) / sinh(d);

    beta = ...
        sinh((lambda * d / 2) / (1 + lambda / 2)) / sinh(d);

    z = alpha * x + beta * a;

    % Correct possible floating-point drift
    z = projectHN(z);

end


%% ------------------------------------------------------------------------
%  Proximal Mapping J_F
% -------------------------------------------------------------------------

function y = JF(x, a, lambda)

    L = lorentz(x, a);

    % Since x,a belong to H^N,
    %
    %       -<x,a>_L >= 1.
    %
    % max is used only as a numerical safeguard.
    argument = max(-L, 1);

    d = acosh(argument);

    % If x and a coincide
    if d < 1e-14
        y = a;
        return;
    end

    % Logarithmic map Log_a(x):
    %
    % Log_a(x)
    % = d_H(a,x)/sqrt(<a,x>_L^2 - 1)
    %   * (x + <a,x>_L a)

    denominator = sqrt(max(L^2 - 1, 0));

    if denominator < 1e-14
        y = a;
        return;
    end

    v = (d / denominator) * (x + L * a);

    % Tangent-space norm
    nv2 = lorentz(v, v);

    nv = sqrt(max(nv2, 0));

    if nv < 1e-14
        y = a;
        return;
    end

    % Unit tangent direction from a toward x
    u = v / nv;

    % Scaling factor
    t = 1 / (1 + lambda);

    % Exponential map
    %
    %       Exp_a(t Log_a(x))
    %
    % IMPORTANT:
    % The sign before sinh must be positive because u points from
    % a toward x.

    y = cosh(t * nv) * a ...
        + sinh(t * nv) * u;

    % Correct possible floating-point drift
    y = projectHN(y);

end


%% ------------------------------------------------------------------------
%  Bifunction F
% -------------------------------------------------------------------------

function value = F(x, y, a)

    % Hyperbolic distances from the reference point a
    dx = dH(x, a);
    dy = dH(y, a);

    % Bifunction:
    %
    % F(x,y) = 1/2 [d_H(y,a)^2 - d_H(x,a)^2]

    value = 0.5 * (dy^2 - dx^2);

end


%% ------------------------------------------------------------------------
%  Hyperbolic Distance
% -------------------------------------------------------------------------

function distance = dH(x, y)

    argument = -lorentz(x, y);

    % Numerical safeguard:
    %
    % theoretically -<x,y>_L >= 1 for x,y in H^N

    argument = max(argument, 1);

    distance = acosh(argument);

end


%% ------------------------------------------------------------------------
%  Lorentz Inner Product
% -------------------------------------------------------------------------

function value = lorentz(u, v)

    % Lorentz inner product on R^(N+1):
    %
    % <u,v>_L
    % = sum_{i=1}^N u_i v_i - u_{N+1} v_{N+1}

    value = dot(u(1:end-1), v(1:end-1)) ...
        - u(end) * v(end);

end


%% ------------------------------------------------------------------------
%  Projection onto the Hyperboloid H^N
% -------------------------------------------------------------------------

function z = projectHN(z)

    lorentz_norm_sq = lorentz(z, z);

    % Points on H^N must satisfy
    %
    %       <z,z>_L = -1.

    if lorentz_norm_sq >= 0
        error('Cannot project a non-timelike vector onto H^N.');
    end

    % Normalize so that <z,z>_L = -1
    z = z / sqrt(-lorentz_norm_sq);

    % Select the upper sheet
    if z(end) < 0
        z = -z;
    end

end