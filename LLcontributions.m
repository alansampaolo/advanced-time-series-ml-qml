function ll_vec = LLcontributions(params, y)

c     = params(1);
phi   = params(2);
kappa = params(3);
alpha = params(4);
beta  = params(5);

T = length(y);

% Construct innovation series ε_t 
eps      = zeros(T,1);
eps(1)   = 0;   % Imposed starting value
for t = 2:T
    eps(t) = y(t) - c - phi*y(t-1);
end

% Construct volatility series σ_t
sigma    = zeros(T,1);
sigma(1) = std(eps);   % Stabilizing initial value
for t = 2:T
    sigma(t) = sqrt(kappa + alpha*eps(t-1)^2 + beta*sigma(t-1)^2);
end

% Remove first observation (imposed assumptions)
eps   = eps(2:end);
sigma = sigma(2:end);

% Conditional log likelihood contributions
ll_vec = -0.5*log(2*pi*sigma.^2) - 0.5*(eps.^2)./(sigma.^2);

end
