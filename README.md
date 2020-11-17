# Madigan

## Aims
This repository contains a framework for conducting experiments within the realm 
of systematic trading. With a focus on statistical arbitrage, the eventual goal
is to create autonomous systems for making trading decisions and executing them.
To this end, robust software is needed to allow for the process of implementing,
validating and deploying ideas, along without the necessary hardware to allow for
running experiments.

## Approach
Hypotheses are to be tested and results aggregated in a directed manner. <br>
Current approach consists of formalizing the trading problem/context in the frame
of reinforcement learning. An agent makes decisions in interacting with
an environment via a defined action space , seeking to
maximize rewards given by the environment. <br>
<br>
***agent -> trader<br>
environment -> 'market', broker, exchange, participants <br>
action space -> buy/sell/desired portfolio <br>
reward -> equity returns / sharpe, transaction costs***<br>
<br>


### Main Components
- Input Space
- Objective / Reward Function
- Core Rl Algorthm
- Function Approximation
#### Input Space
The input space refers to the representation of data which is presented to any
model or agent. This may be a matrix with dimensions corresponding to time, asset
, features etc. Or it could be a flat vector containing the corresponding
compressed information. The choice of history length, features and how to represent relationships between assets should be robust such that



## Timeline



## Installation



## Usage