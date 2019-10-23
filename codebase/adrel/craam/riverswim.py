# Riverswim

import pandas as pd
import numpy as np


def probs_rewards(mdp, a):
    dim = mdp['idstatefrom'].max() + 1

    Pa = np.zeros((dim, dim))
    ra = np.zeros((dim, 1))

    mdp_a = mdp[mdp['idaction'] == a]

    for sample in mdp_a.iterrows():
        state_from = sample[1][0].astype(int)
        state_to = sample[1][2].astype(int)
        prob = sample[1][3]
        reward = sample[1][4]

        Pa[state_from][state_to] = prob
        ra[state_from] = reward
    return Pa, ra


mdp = pd.read_csv('riverswim_mdp.csv')
mdp.index = mdp.index + 1

P, r = probs_rewards(mdp, 0)


# ---------- aggregation -----------------
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
