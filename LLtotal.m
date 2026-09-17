function LL = LLtotal(params, y)

    % Compute individual log-likelihood contributions
    ll_vec = LLcontributions(params, y);

    % Sum contributions to obtain total conditional log likelihood
    LL = sum(ll_vec);

end
