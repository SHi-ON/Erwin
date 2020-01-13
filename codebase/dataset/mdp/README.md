### List of available MDPs

#### 1. **twostate_mdp.csv**: 

MDP from Figure 3.1.1 in Putterman's MDP book, page 34.

**Details**: \
2 states: {s1, s2} -> {0, 1} \
3 actions: {a11, a12, a21} -> {0, 1, 2}


#### 2. **twostate_parametric_mdp.csv**: 

MDP from example 6.4.2 in Putterman's MDP book, page 182.

**Details**: \
2 states: {s1, s2} -> {0, 1} \
2 actions: {a, a2,1} -> {0, 1} \
2 rewards: {$$-a^2$$, -0.5} -> {0, 1} \
3 probabilities: {a/2, 1-a/2, 1} -> {0, 1, 2}


#### 3. **raam_mdp.csv**: 

Three-state deterministic MDP in the [RAAM paper](http://www.cs.unh.edu/~mpetrik/pub/Petrik2014_appendix.pdf).

**Details**: \
3 states: {s1, s2, s3} -> {0, 1, 2} \
3 actions: {a1, a2, 0} -> {0, 1, 2} \
3 rewards: {0, 1, $$\epsilon$$} -> {0, 1, 2} \
3 probabilities: deterministic (all ones) 



