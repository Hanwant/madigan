# Madigan

## Aims
This repository contains a framework for conducting experiments exploring the use of
reinforcement learning in trading finanicla markets. With a focus on statistical arbitrage,
the eventual goal is to create autonomous systems for making trading decisions and executing them.
To this end, robust software is needed to allow for the process of implementing,
validating and deploying ideas, along without the necessary hardware to allow for
running experiments.

## Approach
Hypotheses are to be tested and results aggregated in a directed manner. <br>
Current approach consists of formalizing the trading problem/context in the 
Markov decision process (MDP) framework. An agent makes decisions in interacting with
an environment via a defined action space, seeking to
maximize rewards given by the environment. <br>
<br>
***agent -> trader<br>
environment -> 'market', broker, exchange, participants <br>
action space -> buy/sell/desired portfolio <br>
reward -> equity returns / sharpe / sortino, transaction costs***<br>
<br>


### Main Components
- #### Environment
A suitable formalization and implementation of an environment is required to
create autonomous systems. Should serve the roles of Broker/Exchange and data source.
- Written in C++ with bindings to python - gives peace of mind with respect to speed.
- Currently contains bare minimum functionality for accounting and provides an interface where desired 
number of transaction units can be specified. Order semantics pending.
- #### Objective Function
Objective function and reward shaping for rl should reflect the objectives of a trader. I.e risk-adjusted returns, margin constraints, transaction costs etc
Currently Implemented:
- Raw Log returns
- Naive sharpe and sortino aggregations
- Differential sharpe and sortino updates (DSR & DDR)
- Proximal Portfolio penalty
- #### Input Space
The input space refers to the representation of data which is presented to any
model or agent. This may be a matrix with dimensions corresponding to time, asset
, features etc. Or it could be a flat vector containing the corresponding
compressed information.
- Currently the main 
- #### RL Algorithms
Rl algorithms should be as simple as possible while performing the tasks,
and advanced methods should be continuously considered and tested.
- #### Function Approximation
Given an RL algorithm, a suitable model must be placed as the core agent.
Neural Networks are good general function approximators, and despite high degrees of
freedom, can often generalize well, are composable and provide opportunity for customization.



## Progress - Detailed Components
- Env is written primarily in c++, with python bindings. Components are bare minimum to perform accounting calculations. Core compuational unit is the Portfolio Class, 
it keeps a ledger of positions and performs transactions as well as providing 
risk checking funcitons. Accounts act as containers of portfolios and are wrapped inside a Broker Class which provides parameters for handling transactions such as slippage, transaction cost, etc as well as coordinating the portfolio computations. 
Main components and features have tests - both of accounting logic and of passing
data structures between python / c++.
- The core software framework comprises of the Env, Agent, Preprocessor and 
Trainer Classes. The Env is restricted to being as static as possible, so that 
agents must all interface with it in the same way - by passing a vector of desired
purchases (buy/sell) in units of the indexed assets. Agents with both discrete
and continuous action spaces must translate their model ouputs to assets units
desired for purchase. This allows for a standardized environment.
- The Agent Class contains not just the models being trained but also a reference to the environment (can be queried for accounting info) and the logic required to
train. The interface for training if provided as a generator method which 
periodically yields a list of training metrics to the caller I.e the Trainer class.
The Trainer class co-ordinates training and logging by assembling the env and agent 
components, periodically logging to file, interleaving training with test episodes
and providing a client-server interface (I.e via zmq) for running training jobs.
- Preprocessing - Rollers
- Models -CNNs
- Dash. A dashboard for viewing the results of rl experiments (and training
progress which is periodically logged to file). Made using Qt for Python (PyQt5).
Very important for debugging and interpreting results. Containing graphs of 
training progress (loss, rewards) as well as inidividual test runs. For NN classification tasks, a browser based dashboard using Bokeh is also there).

## To Do
- [X] Train agents to trade Sine Waves 
- [X] Train agents to trade OU Process
- [X] Train agents to trade trending series
- [X] Train agents to trade noisier trending series
- [X] Compose many different series and test multi asset allocation
- [X] Train on synthetic series with multi asset stat arb opportunities
- [X] Train on groups of synthetic series.
- [X] HDFDataSource for market data
- [ ] Order semantics (I.e Market vs Limit/Timed/Stop etc). 
- [ ] Wrap Broker, Account, Portfolio into a backtesting co-ordinator (event driven)
- [ ] Perform backtests and classify agents into a taxonomy (I.e risk profile, 
long/short bias)



## Installation
Requirements: 
- C++ 17 Compiler
- CMake
- Pybind11
- Eigen
- CUDA+CuDNN - if using gpu - recccomended
- Python 3.7 =<
    - Pytorch 
    - Numpy
    - Pandas
    - HDF5
    - PyQt5 <br>
    
Install python Library via: <br>
    ```
    python setup.py install
    ```<br>
    or<br>
    ``` 
    pip install .
    ```<br>


## Usage
