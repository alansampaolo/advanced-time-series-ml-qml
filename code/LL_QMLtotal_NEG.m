function nLL = LL_QMLtotal_NEG(params, y)

    % QML log-likelihood contributions
    ll_vec = LL_QMLcontributions(params, y);

    % Negative log-likelihood (required for minimization)
    nLL = -sum(ll_vec);

end
