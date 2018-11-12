data {                          
int<lower=0> N;                # number of observations
int<lower=0,upper=1> detected[N];  # setting the dependent variable (detected or not) as binary
vector[N] bio10_5;             # independent variable 1
vector[N] bio10_prMay;         # independent variable 2
}
parameters {
real alpha;                    # intercept
real beta_bio10_5;             # beta for bio10_5
real beta_bio10_prMay;         # beta for bio10_prMay
}
model {
alpha ~ normal(0,10);         # you can set priors for all betas
beta_bio10_5 ~ normal(0,10);     # if you prefer not to, uniform priors will be used
beta_bio10_prMay ~ normal(0,10);
detected ~ bernoulli_logit(alpha +beta_bio10_prMay * bio10_5 + beta_bio10_5 * bio10_prMay); # model
}
generated quantities {         # simulate quantities of interest
real y_hat;                    # create a new variable for the predicted values
y_hat <- inv_logit(alpha + beta_bio10_5 * (-0.89) + beta_bio10_prMay * (-2.28)); # model
}