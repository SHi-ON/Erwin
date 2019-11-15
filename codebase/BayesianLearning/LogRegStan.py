import sys
import glob

import pystan
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline
# %precision 4
plt.style.use('ggplot')

df = pd.read_csv('../dataset/HtWt.csv')
df.head()  # printing first 5 rows

log_reg_code = """
data {
    int<lower=0> n;
    int male[n];
    real weight[n];
    real height[n];
}
transformed data {}
parameters {
    real a;
    real b;
    real c;
}
transformed parameters {}
model {
    a ~ normal(0, 10);
    b ~ normal(0, 10);
    c ~ normal(0, 10);
    for(i in 1:n) {
        male[i] ~ bernoulli(inv_logit(a*weight[i] + b*height[i] + c));
  }
}
generated quantities {}
"""

log_reg_dat = {
             'n': len(df),
             'male': df.male,
             'height': df.height,
             'weight': df.weight
            }

fit = pystan.stan(model_code=log_reg_code, data=log_reg_dat, iter=2000, chains=1)