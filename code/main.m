%% ============================================================
%                 ATS Mandatory Assignment – Main Script
% This script:
% 1. Simulates the AR(1)-SV process
% 2. Computes conditional log-likelihoods
% 3. Estimates parameters via ML and QML
% 4. Performs inference and large-sample checks
% 5. Runs Monte Carlo comparison ML vs QML
% ============================================================

clear; clc; close all;
%% TASK 1 — Simulation
disp(' ')
disp('***************   TASK 1   ***************')
disp(' ')

rng('default');
rng(123);   % Ensure reproducibility

% --- Model parameters ---
T_sim  = 800;
c      = 4;
phi    = 0.9;
kappa  = 0.75;
alpha  = 0.3;
beta   = 0.6;

y0 = c / (1 - phi);   % Stationary mean as initial value

% --- Simulate stochastic volatility process ---
[y_raw, sigma_raw] = simulateSV(T_sim, c, phi, kappa, alpha, beta, y0);

% --- Burn-in removal ---
burn_in = 50;
y     = y_raw(burn_in+1:end);
sigma = sigma_raw(burn_in+1:end);

% --- Plot simulated series ---
figure('Color','w');
plot(y,'LineWidth',1.2);
grid on;
title('Simulated AR(1) Process with Stochastic Volatility','FontSize',14);
xlabel('Time','FontSize',12);
ylabel('y_t','FontSize',12);
set(gca,'FontSize',12);



%% TASK 2 — Conditional Log Likelihood Evaluation
disp(' ')
disp('***************   TASK 2   ***************')

% Two alternative parameter sets
params_a = [4; 0.95; 0.75; 0.3; 0.6];
params_b = [1.5; 0.75; 0.5; 0.6; 0.1];

% Compute log-likelihood values
LL_a = LLtotal(params_a, y);
LL_b = LLtotal(params_b, y);

% Display results
fprintf('\n%-12s %-18s\n','Parameter Set','Log-Likelihood')
fprintf('----------------------------------------\n')
fprintf('%-12s %-18.6f\n','(a)',LL_a)
fprintf('%-12s %-18.6f\n','(b)',LL_b)



%% TASK 3 — ML Estimation (T = 750)
disp(' ')
disp('***************   TASK 3   ***************')
disp(' ')

% addpath("CML")   

% --- Starting values ---
theta0 = [3; 0.5; 0.5; 0.1; 0.5];

% --- Optimization settings ---
options = optimset('Display','off','TolX',1e-40,'TolFun',1e-40,...
                   'MaxIter',1000,'MaxFunEvals',100000);

algorithm = 1;   % fminsearch
covPar    = 1;   % Hessian-based covariance matrix

% --- ML estimation ---
[x, f, g, cov, retcode] = CML(@LLtotal_NEG, @LLcontributions,...
                              y, theta0, algorithm, covPar, options);

param_names = {'c','phi','kappa','alpha','beta'};

% --- Display estimates ---
disp('==================== ML RESULTS FOR T = 750 =====================')
disp(' ')
fprintf('%-10s %-14s\n','Parameter','Estimate')
fprintf('--------------------------------------\n')
for i = 1:length(x)
    fprintf('%-10s %10.6f\n', param_names{i}, x(i));
end

% --- Standard errors and confidence intervals ---
SE = sqrt(diag(cov));
z = 1.96;
CI_lower = x - z * SE;
CI_upper = x + z * SE;

disp(' ')
disp('----------------- ESTIMATES, SE, CONFIDENCE INTERVALS -----------------')
fprintf('%-10s %-12s %-10s %-12s %-12s\n','Param','Estimate','SE','CI_low','CI_high')
fprintf('-----------------------------------------------------------------------\n')

for i = 1:length(x)
    fprintf('%-10s %10.6f %10.6f %12.6f %12.6f\n',...
        param_names{i}, x(i), SE(i), CI_lower(i), CI_upper(i));
end

% --- t-test for H0: phi = 0.8 ---
disp(' ')
disp('--------------------------- T-TEST: H0 : phi = 0.8 ---------------------------')

phi0 = 0.8;
phi_hat = x(2);
SE_phi  = SE(2);
t_stat = (phi_hat - phi0)/SE_phi;
p_value = 2*(1 - normcdf(abs(t_stat)));

fprintf('%-20s %12.6f\n','phi_hat',phi_hat)
fprintf('%-20s %12.6f\n','SE(phi_hat)',SE_phi)
fprintf('%-20s %12.6f\n','t-statistic',t_stat)
fprintf('%-20s %12.6f\n','p-value',p_value)

if abs(t_stat) > 1.96
    disp('Decision: Reject H0 at 5% level.')
else
    disp('Decision: Do NOT reject H0 at 5% level.')
end



%% ML for Large Sample (T = 49,950)
disp(' ')
disp('==================== ML RESULTS FOR T = 49,950 =====================')
disp(' ')

% --- True parameters used for re-simulation ---
c_true     = 4;
phi_true   = 0.9;
kappa_true = 0.75;
alpha_true = 0.3;
beta_true  = 0.6;

T_large = 50000;

rng(123);
y0 = c_true/(1 - phi_true);

% --- Generate long sample ---
[y_raw_large, ~] = simulateSV(T_large, c_true, phi_true,...
                              kappa_true, alpha_true, beta_true, y0);

burn_in = 50;
y_large = y_raw_large(burn_in+1:end);

% --- ML estimation ---
[x_large, f_large, ~, cov_large, ~] = CML(@LLtotal_NEG,...
    @LLcontributions, y_large, theta0, algorithm, covPar, options);

% CI
SE_large = sqrt(diag(cov_large));
CI_low_large  = x_large - z*SE_large;
CI_high_large = x_large + z*SE_large;

fprintf('%-10s %-12s %-10s %-12s %-12s\n','Param','Estimate','SE','CI_low','CI_high')
fprintf('-----------------------------------------------------------------------\n')

for i = 1:length(x_large)
    fprintf('%-10s %10.6f %10.6f %12.6f %12.6f\n',...
        param_names{i}, x_large(i), SE_large(i), CI_low_large(i), CI_high_large(i));
end

% --- t-test ---
disp(' ')
disp('--------------------------- T-TEST: H0 : phi = 0.8 ---------------------------')

phi_hat_large = x_large(2);
SE_phi_large  = SE_large(2);

t_stat_large = (phi_hat_large - phi0) / SE_phi_large;
p_large      = 2 * (1 - normcdf(abs(t_stat_large)));

fprintf('%-22s %12.6f\n','phi_hat (large)',phi_hat_large)
fprintf('%-22s %12.6f\n','SE(phi_hat)',SE_phi_large)
fprintf('%-22s %12.6f\n','t-statistic',t_stat_large)
fprintf('%-22s %12.6f\n','p-value',p_large)

if abs(t_stat_large) > 1.96
    disp('Decision: Reject H0 at 5% level.')
else
    disp('Decision: Do NOT reject H0 at 5% level.')
end



%% TASK 4 — QML Estimation
disp(' ')
disp('***************   TASK 4   ***************')
disp(' ')

% --- QML starting values ---
theta0_qml = [3; 0.5; 1];
covPar_qml = 3;   % Robust covariance

% --- QML for both sample sizes ---
[x_qml,       ~, ~, cov_qml,       ~] = CML(@LL_QMLtotal_NEG, @LL_QMLcontributions,...
                                            y,       theta0_qml, algorithm, covPar_qml, options);

[x_qml_large, ~, ~, cov_qml_large, ~] = CML(@LL_QMLtotal_NEG, @LL_QMLcontributions,...
                                            y_large, theta0_qml, algorithm, covPar_qml, options);

param_names_qml = {'c','phi','sigma'};

disp('==================== QML RESULTS FOR T = 750 =====================')
disp(' ')

% --- Confidence intervals ---
SE_qml = sqrt(diag(cov_qml));
CI_low_qml  = x_qml - z*SE_qml;
CI_high_qml = x_qml + z*SE_qml;

fprintf('%-10s %-12s %-10s %-12s %-12s\n','Param','Estimate','SE','CI_low','CI_high')
fprintf('-----------------------------------------------------------------------\n')

for i = 1:length(x_qml)
    fprintf('%-10s %10.6f %10.6f %12.6f %12.6f\n',...
        param_names_qml{i}, x_qml(i), SE_qml(i), CI_low_qml(i), CI_high_qml(i));
end

% --- t-test ---
disp(' ')
disp('--------------------------- T-TEST: H0 : phi = 0.8 ---------------------------')

phi_hat = x_qml(2);
SE_phi  = SE_qml(2);

t_stat = (phi_hat - phi0) / SE_phi;
p_value = 2*(1 - normcdf(abs(t_stat)));

fprintf('%-22s %12.6f\n','phi_hat',phi_hat)
fprintf('%-22s %12.6f\n','SE(phi_hat)',SE_phi)
fprintf('%-22s %12.6f\n','t-statistic',t_stat)
fprintf('%-22s %12.6f\n','p-value',p_value)

if abs(t_stat) > 1.96
    disp('Decision: Reject H0 at 5% level.')
else
    disp('Decision: Do NOT reject H0 at 5% level.')
end



%% QML for Large Sample
disp(' ')
disp('==================== QML RESULTS FOR T = 49,950 =====================')
disp(' ')

SE_qml_large = sqrt(diag(cov_qml_large));
CI_low_large = x_qml_large - z*SE_qml_large;
CI_high_large = x_qml_large + z*SE_qml_large;

fprintf('%-10s %-12s %-10s %-12s %-12s\n','Param','Estimate','SE','CI_low','CI_high')
fprintf('-----------------------------------------------------------------------\n')

for i = 1:length(x_qml_large)
    fprintf('%-10s %10.6f %10.6f %12.6f %12.6f\n',...
        param_names_qml{i}, x_qml_large(i), SE_qml_large(i),...
        CI_low_large(i), CI_high_large(i));
end

% --- t-test ---
disp(' ')
disp('--------------------------- T-TEST: H0 : phi = 0.8 ---------------------------')

phi_hat_L = x_qml_large(2);
SE_phi_L  = SE_qml_large(2);

t_stat_L = (phi_hat_L - phi0) / SE_phi_L;
p_L      = 2 * (1 - normcdf(abs(t_stat_L)));

fprintf('%-22s %12.6f\n','phi_hat (large)',phi_hat_L)
fprintf('%-22s %12.6f\n','SE(phi_hat)',SE_phi_L)
fprintf('%-22s %12.6f\n','t-statistic',t_stat_L)
fprintf('%-22s %12.6f\n','p-value',p_L)

if abs(t_stat_L) > 1.96
    disp('Decision: Reject H0 at 5% level.')
else
    disp('Decision: Do NOT reject H0 at 5% level.')
end



%% TASK 5 — Monte Carlo Comparison ML vs QML
disp(' ')
disp('***************   TASK 5   ***************')
disp(' ')

rng(123);   % Reproducibility

K = 500;
T_sim = 800;
burn_in = 50;

% Matrices for storing estimates
ML_estimates  = nan(K,5);
QML_estimates = nan(K,3);

theta0_ML  = [3; 0.5; 0.5; 0.1; 0.5];
theta0_QML = [3; 0.5; 1];

% Suppress warnings triggered by occasional bad draws
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:illConditionedMatrix');

for k = 1:K

    % --- Simulate one realization ---
    [y_raw_MC, ~] = simulateSV(T_sim, c_true, phi_true,...
                               kappa_true, alpha_true, beta_true, y0);
    y_MC = y_raw_MC(burn_in+1:end);

    % --- ML estimation with check for singularities ---
    lastwarn('');
    [x_ML, ~, ~, cov_ML_k, ret_ML] = ...
        CML(@LLtotal_NEG, @LLcontributions, y_MC,...
            theta0_ML, algorithm, 1, options);

    [msg, ~] = lastwarn;
    if ret_ML == 1 && ~contains(msg,'singular') && ~contains(msg,'ill')
        ML_estimates(k,:) = x_ML';
    end

    % --- QML estimation with same stability check ---
    lastwarn('');
    [x_QML, ~, ~, cov_QML_k, ret_QML] = ...
        CML(@LL_QMLtotal_NEG, @LL_QMLcontributions, y_MC,...
            theta0_QML, algorithm, 3, options);

    [msg, ~] = lastwarn;
    if ret_QML == 1 && ~contains(msg,'singular') && ~contains(msg,'ill')
        QML_estimates(k,:) = x_QML';
    end

end

% Extract ML/QML estimates of c and phi
ML_c   = ML_estimates(:,1);
ML_phi = ML_estimates(:,2);
QML_c  = QML_estimates(:,1);
QML_phi= QML_estimates(:,2);

% --- Plot kernel densities ---
figure('Color','w');

subplot(2,1,1)
hold on
[f_ML_c, x_ML_c] = ksdensity(ML_c);
[f_QML_c, x_QML_c] = ksdensity(QML_c);
plot(x_ML_c, f_ML_c,'LineWidth',2)
plot(x_QML_c, f_QML_c,'LineWidth',2)
xline(c_true,'k--','LineWidth',1.5);   % True parameter
xlabel('c','FontSize',12)
ylabel('Density','FontSize',12)
title('Kernel Density of ML and QML Estimates for c','FontSize',14)
legend('ML','QML','True value')
grid on
hold off

subplot(2,1,2)
hold on
[f_ML_phi, x_ML_phi] = ksdensity(ML_phi);
[f_QML_phi, x_QML_phi] = ksdensity(QML_phi);
plot(x_ML_phi, f_ML_phi,'LineWidth',2)
plot(x_QML_phi, f_QML_phi,'LineWidth',2)
xline(phi_true,'k--','LineWidth',1.5);
xlabel('\phi','FontSize',12)
ylabel('Density','FontSize',12)
title('Kernel Density of ML and QML Estimates for \phi','FontSize',14)
legend('ML','QML','True value')
grid on
hold off

disp(' ')
disp('***************   END OF ALL TASKS   ***************')
disp(' ')
