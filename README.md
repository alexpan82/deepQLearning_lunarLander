# Solving Lunar Lander using a Deep Q Network
## Q-learning implemented from scratch
This code base has 3 main functionalities:
* Train an agent using DQN to solve Lunar Lander
* Implement trained agent on the environment
* Plot training and performance results

Methods and implementation based on Mnih et al: https://www.nature.com/articles/nature14236

External Python packages used:
* matplotlib v=3.3.4
* numpy v=1.19.5
* tensorflow v=2.4.1
  * Supports both CPU-only and GPU versions
* gym v=0.17.2
* box2d-py

# Example usage
## 1) Train agent:
* Renders plots and game in real time
* Parameters may be manually changed on line 10 of main.py
* Outputs 2 files
  * q_model.h5: h5 file with stored neural network parameters
  * metadata.npy: Binary file containing metadata stored from training
```
python main.py train
```
## 2) Implement agent
* By default, imports ./q_model.h5 and runs 100 consecutive episodes
```
python main.py play
```
## 3) Plot metadata
```
python plotter.py --help
Example:
python plotter.py --output epsilon.png --legend 1 --npy ./*/metadata.npy --avg 50
```
## 4) Explanation of other files
* dqn.py: Implements the Q-learning side of DQN
* neuralNet.py: Implements the predictive and target networks
* test_initial_states.npy: A set of 50 random initial states from Lunar Lander environment. Only used to plot predicted Q-values.
