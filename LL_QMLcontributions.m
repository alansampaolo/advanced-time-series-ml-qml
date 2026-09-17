function ll_vec = LL_QMLcontributions(params, y)

c     = params(1);
phi   = params(2);
sigma = params(3);     % Constant standard deviation (QML)

T = length(y);

% Construct innovation series ε_t 
eps      = zeros(T,1);
eps(1)   = 0;   % Imposed starting value
for t = 2:T
    eps(t) = y(t) - c - phi*y(t-1);
end

% Remove first imposed value 
eps = eps(2:end);

% QML log likelihood contributions
ll_vec = -0.5*log(2*pi*sigma^2) - 0.5*(eps.^2)/(sigma^2);

end
