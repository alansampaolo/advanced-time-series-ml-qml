function [y, sigma] = simulateSV(T, c, phi, kappa, alpha, beta, y0)

y     = zeros(T,1);
sigma = zeros(T,1);
eps_t = zeros(T,1);

% Initialization
y(1) = y0;                                 % Starting point for AR(1)
sigma2_uncond = kappa / (1 - alpha - beta); % Unconditional variance of SV
sigma(1) = sqrt(sigma2_uncond);             % Initial volatility

% Simulation loop 
for t = 2:T

    z_t = randn;   % Standard normal innovation

    % Update stochastic volatility using SV recursion
    sigma(t) = sqrt( kappa + alpha*eps_t(t-1)^2 + beta*sigma(t-1)^2 );

    % Shock for the AR(1) model
    eps_t(t) = sigma(t) * z_t;

    % AR(1) update
    y(t) = c + phi*y(t-1) + eps_t(t);
end

end
