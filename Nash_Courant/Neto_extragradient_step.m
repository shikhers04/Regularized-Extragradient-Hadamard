function [x_k, K, Err] = Neto_extragradient_step( ...
    F, d, lb, ub, x0, eta, max_inner, tol_Neto)
%%% NETO_EXTRAGRADIENT_STEP Solves an equilibrium problem using
%%% an extragradient-type algorithm.
%
% The method computes
%
%   y_k = argmin_{y in K} {
%             F(x_k,y) + d(x_k,y)^2/(2*eta_k)
%         },
%
%   x_{k+1} = argmin_{y in K} {
%             F(y_k,y) + d(x_k,y)^2/(2*eta_k)
%         }.
%
% INPUTS
%   F          : Bifunction handle F(x,y), returning a scalar
%   d          : Distance handle d(x,y), returning the distance
%   A,b_cons   : Linear constraints A*x <= b_cons
%   x0         : Initial point
%   eta        : Positive scalar or function handle @(k)
%   max_inner  : Maximum number of iterations
%   tol_Neto   : Stopping tolerance
%
% OUTPUTS
%   x_k        : Approximate solution
%   K          : Number of performed iterations
%   Err        : Final error

    %% Initialization
    x_k = x0(:);

    %% fmincon options
    options = optimoptions('fmincon', ...
        'Algorithm', 'interior-point', ...
        'Display', 'off', ...
        'OptimalityTolerance', 1e-10, ...
        'StepTolerance', 1e-12, ...
        'ConstraintTolerance', 1e-10, ...
        'MaxIterations', 1000, ...
        'MaxFunctionEvaluations', 10000);

    %% Main iterations
    for k = 1:max_inner

        x_k_prev = x_k;

        %% Choose eta_k
        if isa(eta, 'function_handle')
            eta_k = eta(k);
        else
            eta_k = eta;
        end

        if ~isscalar(eta_k) || eta_k <= 0
            error('eta_k must be a positive scalar.');
        end

        %% Step 1
        % y_k = argmin_y {
        %          F(x_k,y) + d(x_k,y)^2/(2*eta_k)
        %       }

        fun1 = @(y) F(x_k_prev, y) ...
            + (1 / (2 * eta_k)) * d(x_k_prev, y)^2;

        y_k = fmincon(fun1, x_k_prev, ...
            [], [], [], [], lb, ub, [], options);

        y_k = y_k(:);

        %% Step 2
        % x_{k+1} = argmin_y {
        %              F(y_k,y) + d(x_k,y)^2/(2*eta_k)
        %           }

        fun2 = @(y) F(y_k, y) ...
            + (1 / (2 * eta_k)) * d(x_k_prev, y)^2;

        x_k = fmincon(fun2, y_k, ...
            [], [], [], [], lb, ub, [], options);

        x_k = x_k(:);

        %% Error
        Err = d(x_k_prev, x_k);

        %% Stopping condition
        if Err <= tol_Neto
            break;
        end
    end

    %% Number of iterations
    K = k;

end