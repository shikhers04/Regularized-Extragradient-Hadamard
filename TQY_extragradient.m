function [x_k, K, Err] = TQY_extragradient( ...
    F_n, d, lb, ub, x_n, tau, delta, chi, xi, sigma, ...
    max_inner, tol_TQY)
% TQY_EXTRAGRADIENT performs the TQY extragradient method.
%
% INPUTS:
%   F_n        - Bifunction F_n(x,y)
%   d          - Distance function handle
%   A,b_cons   - Linear constraints A*x <= b_cons
%   x_n        - Initial iterate
%   tau        - Initial step size tau_0
%   delta      - Algorithm parameter
%   chi        - Algorithm parameter
%   xi         - Positive scalar or function handle @(k)
%   sigma      - Positive scalar or function handle @(k)
%   max_inner  - Maximum number of iterations
%   tol_TQY    - Stopping tolerance
%
% OUTPUTS:
%   x_k        - Approximate solution
%   K          - Number of performed iterations
%   Err        - Error history

    %% Initialization

    x_k   = x_n(:);
    tau_k = tau;

    Err = zeros(max_inner,1);

    %% Optimization options

    options = optimoptions('fmincon', ...
        'Algorithm','interior-point', ...
        'Display','off', ...
        'OptimalityTolerance',1e-10, ...
        'StepTolerance',1e-12, ...
        'ConstraintTolerance',1e-10, ...
        'MaxIterations',1000, ...
        'MaxFunctionEvaluations',10000);

    %% Main iterations

    for k = 1:max_inner

        x_k_prev = x_k;

        %% Parameters xi_k and sigma_k

        if isa(xi,'function_handle')
            xi_k = xi(k);
        else
            xi_k = xi;
        end

        if isa(sigma,'function_handle')
            sigma_k = sigma(k);
        else
            sigma_k = sigma;
        end

        %% Step 1

        fun1 = @(u) ...
            F_n(x_k_prev,u) ...
            + (1/(2*tau_k)) * d(x_k_prev,u)^2;

        y_k = fmincon( ...
            fun1, x_k_prev, ...
            [], [], ...
            [], [], lb, ub, [], options);

        y_k = y_k(:);

        %% Step 2

        fun2 = @(v) ...
            F_n(y_k,v) ...
            + (1/(2*chi*tau_k)) * d(x_k_prev,v)^2;

        x_k = fmincon( ...
            fun2, y_k, ...
            [], [], ...
            [], [], lb, ub, [], options);

        x_k = x_k(:);

        %% Step 3: Update tau

        Delta_k = ...
            F_n(x_k_prev,x_k) ...
            - F_n(x_k_prev,y_k) ...
            - F_n(y_k,x_k);

        if Delta_k > 0

            tau_next = min( ...
                delta * d(x_k_prev,y_k) * d(x_k,y_k) / Delta_k, ...
                xi_k * tau_k + sigma_k);

        else

            tau_next = xi_k * tau_k + sigma_k;

        end

        %% Convergence error

        Err(k) = d(x_k_prev,x_k);

        if Err(k) <= tol_TQY
            break;
        end

        %% Update step size

        tau_k = tau_next;

    end

    %% Outputs

    K   = k;
    Err = Err(1:K);

end