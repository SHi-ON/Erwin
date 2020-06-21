### List of available MDPs

### 1. **Two-State** 

MDP from Figure 3.1.1 in Putterman's MDP book, page 34.

file name: `twostate_mdp.csv`

**Details**: \
2 states: {s1, s2} -> {0, 1} \
3 actions: {a11, a12, a21} -> {0, 1, 2}


### 2. **Two-State Parametric** 

MDP from example 6.4.2 in Putterman's MDP book, page 182.

file name: `twostate_parametric_mdp.csv`

**Details**: \
2 states: {s1, s2} -> {0, 1} \
2 actions: {a, a2,1} -> {0, 1} \
2 rewards: {$$-a^2$$, -0.5} -> {0, 1} \
3 probabilities: {a/2, 1-a/2, 1} -> {0, 1, 2}


### 3. **RAAM** 

Three-state deterministic MDP problem from the [RAAM paper](http://www.cs.unh.edu/~mpetrik/pub/Petrik2014_appendix.pdf).

file name: `raam_mdp.csv`

**Details**: \
3 states: {s1, s2, s3} -> {0, 1, 2} \
3 actions: {a1, a2, 0} -> {0, 1, 2} \
3 rewards: {0, 1, $$\epsilon$$} -> {0, 1, 2} \
3 probabilities: deterministic (all ones) 


### 4. **Machine Replacement** 

Machine Replacement MDP problem from the [Percentile Optimization paper](http://web.hec.ca/pages/erick.delage/percentileMDP.pdf), Figure 3.

file name: `machine_replacement_mdp.csv`

**Details**: \
10 states: {1, 2, ..., 8, R1, R2} -> {0, 1, 2, ..., 9} \
2 actions: either **"do nothing"**=0 or **"repair"**=1


### 5. **RiverSwim** 

RiverSwim MDP problem from [Strehl et al. 2004](http://web.hec.ca/pages/erick.delage/percentileMDP.pdf), Figure 3.

file name: `river_swim_mdp.csv`

**Details**: \
6 states: {0, 1, 2, ..., 5} \
2 actions: **left**=0 and **right**=1


### 6. **SixArms** 

SixArms MDP problem from [Strehl et al. 2004](http://web.hec.ca/pages/erick.delage/percentileMDP.pdf), Figure 3.

file name: `six_arms_mdp.csv`

**Details**: \
7 states: {0, 1, 2, ..., 6} \
6 actions: {0, 1, 2, ..., 5}




