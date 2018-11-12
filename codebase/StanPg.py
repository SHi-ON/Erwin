import pystan
import numpy as np
import matplotlib.pyplot as plt

stan_code_model = """
data {
    int ns;  // num of samples
    int nf;  // num of features
    matrix[ns,nf] phi;  // feature matrix
    vector[ns] y;  // response
}
parameters {
    vector[nf] beta;
    real<lower=0> epsilon; 
}
model {
    y ~ normal(phi * beta, epsilon);
}
"""


def distro():
    # stan_data = {'num_samples': 30, 'y': d}

    randNumGen = np.random.RandomState(7)
    n_samp = 10
    n_feat = 4

    stan_model = pystan.StanModel(model_code=stan_code_model)

    features = randNumGen.normal(0, 1, size=(n_samp, n_feat))
    response = features @ np.array([-1, 0.4, 0, 0]) + randNumGen.normal(scale=0.2, size=n_samp)

    stan_fit = stan_model.sampling(data={'ns': n_samp,
                                         'nf': n_feat,
                                         'phi': features,
                                         'y': response},
                                   iter=1000, chains=4, warmup=500, n_jobs=-1, seed=7)

    chains = stan_fit.extract()

    chains['beta'].shape

    ft_plot =  stan_fit.plot()
    stan_fit.plot()


    lin.plot(chains['beta'][:,0])



def main():
    distro()


if __name__ == '__name__':
    main()
