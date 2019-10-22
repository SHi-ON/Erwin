import numpy as np

# import matplotlib.pyplot as

P = np.array([[0.2, 0.3, 0.5],
              [0.1, 0.6, 0.3],
              [0.9, 0.1, 0.0]])



r = np.array([-10, 7, 3])

gamma = 0.99

Phi = np.array([[1, 0], [1, 0], [0, 1]])
print(Phi)
D = np.diag([0.5, 0.5, 1])

v = np.linalg.solve((np.eye(3) - gamma * P), r)

# LSTD
# weights
w = np.linalg.solve(Phi.T @ D @ Phi - gamma * Phi.T @ D @ P @ Phi, Phi.T @ D @ r)

# Value function
print(Phi @ w)

###
### Riverswim

import pandas as pd

mdp = pd.read_csv('riverswim_mdp.csv')
mdp.index = mdp.index + 1

mdp['probability']

aggregation = pd.DataFrame({'idstate': range(6),
                            'idstate_agg': [0] + list(range(5)),
                            'weights': [0.5] * 2 + [1] * 4},
                           index=range(1, 7))

joined_from = mdp.merge(aggregation,
                        left_on='idstatefrom',
                        right_on='idstate',
                        how='inner') \
    .drop(columns=['idstate'])

joined_from.index = joined_from.index + 1
joined_from['idstatefrom'] = joined_from['idstate_agg']
joined_from.drop(columns='idstate_agg', inplace=True)

joined_to = joined_from.merge(aggregation.drop(columns='weights'),
                              left_on='idstateto',
                              right_on='idstate') \
    .drop(columns=['idstate'])
joined_to.index = joined_to.index + 1
joined_to['idstateto'] = joined_to['idstate_agg']
joined_to.drop(columns='idstate_agg', inplace=True)

joined_to['probability'] = joined_to['probability'] * joined_to['weights']
joined_to['reward'] = joined_to['reward'] * joined_to['weights']

dd = joined_to.groupby(by=['idstatefrom', 'idaction', 'idstateto']).agg({'probability': 'sum', 'reward': 'sum'})

mdp_agg = pd.DataFrame(dd.reset_index())
mdp_agg.index = mdp_agg.index + 1

mdp_agg

