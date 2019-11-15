import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

import pystan

df = pd.read_csv("../dataset/barberry_sim_NewEngland.csv")
num_sub_samples = 200
sub_samples = df.sample(n=num_sub_samples)

abundance_samples = pd.DataFrame()
for index, row in sub_samples.iterrows():
    abundance = max(row['N'], 1)  # sub_samples.loc[index, 'N']
    if abundance > 1:
        sub_samples.loc[index, 'N'] = 1
    abundance_samples = abundance_samples.append([sub_samples.loc[index]] * 1)  # int(abundance)

X = abundance_samples[['bio10_5', 'bio10_prMay']].values
Y = abundance_samples['N'].values


### Bayesian Logistic Regression model using Stan
def fit_bayesian_logistic_regression(X, y):
    stan_dat = {'N': len(X), 'detected': y.astype(int), 'bio10_5': X[:, 0], 'bio10_prMay': X[:, 1]}

    sm = pystan.StanModel(file='./models/model_logistic.stan')  # pystan.StanModel(model_code=m1)
    # fit = sm.sampling(data=stan_dat, iter=20, chains=4, refresh=0)
    fit = sm.sampling(data=stan_dat, iter=180, chains=3, refresh=0)

    mean_alpha, std_alpha = np.mean(fit.extract()['alpha']), np.std(fit.extract()['alpha'])
    mean_beta_bio10_5, std_beta_bio10_5 = np.mean(fit.extract()['beta_bio10_5']), np.std(fit.extract()['beta_bio10_5'])
    mean_beta_bio10_prMay, std_beta_bio10_prMay = np.mean(fit.extract()['beta_bio10_prMay']), np.std(
        fit.extract()['beta_bio10_prMay'])

    return mean_alpha, mean_beta_bio10_5, mean_beta_bio10_prMay, std_alpha, std_beta_bio10_5, std_beta_bio10_prMay


def inv_logit(p):
    # with precision(100000):
    return round(math.exp(p) / (1 + math.exp(p)))


### Fit logistic regression model with Scikit-Learn & Stan
freq_accuracy, bayes_accuracy = [], []
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
kf = model_selection.KFold(n_splits=2)
for train_index, test_index in kf.split(abundance_samples):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # fit frequentist model with sklearn
    logfit = logreg.fit(X_train, y_train)
    res = logfit.predict(X_test)
    freq_accuracy.append(1 - np.mean(abs(res - y_test)))

    # fit bayesian model with Stan
    alpha, beta1, beta2, std_alpha, std_beta1, std_beta2 = fit_bayesian_logistic_regression(X_train, y_train)
    res = list(map(inv_logit, alpha + np.dot(X_test, np.array([beta1, beta2]))))
    bayes_accuracy.append(1 - np.mean(abs(np.array(res) - y_test)))
    break

print("freq_accuracy", freq_accuracy, "bayes_accuracy", bayes_accuracy)

###
import matplotlib.mlab as mlab

x = np.linspace(alpha - 3 * std_alpha, alpha + 3 * std_alpha, 100)
plt.plot(x, mlab.normpdf(x, alpha, std_alpha))
plt.title("Weight distribution for feature: alpha/intercept")
plt.show()

###
x = np.linspace(beta1 - 3 * std_beta1, beta1 + 3 * std_beta1, 100)
plt.plot(x, mlab.normpdf(x, beta1, std_beta1))
plt.title("Weight distribution for feature: bio10_5")
plt.show()


x = np.linspace(beta2 - 3 * std_beta2, beta2 + 3 * std_beta2, 100)
plt.plot(x, mlab.normpdf(x, beta2, std_beta2))
plt.title("Weight distribution for feature: bio10_prMay")
plt.show()
