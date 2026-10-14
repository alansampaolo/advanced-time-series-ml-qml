function nLL = LLtotal_NEG(params, y)

    % Log-likelihood contributions
    ll_vec = LLcontributions(params, y);

    % Negative log-likelihood (needed because CML minimizes)
    nLL = -sum(ll_vec);

end
