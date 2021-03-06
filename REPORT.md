## Project Report
### Learning Algorithm
This project uses the algorithm Proximal Policy Optimization (PPO) where we aim to maximize the surrogate objective:
objective = E [ r(θ) * A^t]<br> 
    where r(θ) =  π(a_t|s_t) / π_old(a_t|s_t)<br> 

To avoid an excessively large policy update, PPO uses a clipped surrogate objective that clips the objective in respect to r(θ) in the range of (1-Ɛ, 1+Ɛ). We take the mininum of the original objective and the clipped objective. 

The objective is further augmented by introducing an entropy bonus that ensures sufficient exploration. Combining these terms, the following objective is obtained:<br> 
Polic Loss: L_CLIP(θ) = min[ r(θ), clip ( r(θ), 1-Ɛ, 1+Ɛ )*A ] - β * entropy<br> 
Value Loss: L_VF(θ) = ( Vθ(s_t) - V_t_target )^2<br> 

The PPO algorithm implemented is Actor-Critic style, with a Actor (Policy) network and a Critic (Value) network where the actor outputs the action probability distribution given the states, and the critic network outputs an estimated value of the state.

#### Model Architecture
The PPO algorithm implemented is Actor-Critic style, with a Actor (Policy) network and a Critic (Value) network sharing the same architecture (except for the output size). <br> 
##### Actor Network <br> 
* Input layer is equal to the length of states vector = 33<br> 
* First hidden layer consists of 256 nodes<br> 
* Second hidden layer consists of 256 nodes<br> 
* Output layer is equal to the length of actions vector = 4<br> 

##### Critic Network <br> 
* Input layer is equal to the length of states vector = 33
* First hidden layer consists of 256 nodes
* Second hidden layer consists of 256 nodes
* Output layer 1


#### Hyperparameters
##### Interactions
Discount rate (GAMMA) = 0.99<br> 
Rollout length (ROLLOUT_LENGTH) = 1024<br> 

##### Learning Process
PPO clip (CLIP) = 0.2<br> 
Entropy coefficient (BETA) = 0.01<br> 

##### Optimization
Number of epochs for optimization (NUM_EPOCHS) = 10<br> 
Gradient clip for optimization (GRADIENT_CLIP) = 5<br> 
Adam Learning Rate (LR) = 3e-4<br> 
Adam epsilong (EPSILON) = 1e-5<br> 


### Plot of Rewards

![Alt Text](https://github.com/bonniesjli/PPO_Reacher_UnityML/blob/master/asset/PPO.png)
The agent was able to solve the environment in 224 episodes. 

### Ideas for Future Work
* Hyperparameter optimization
- One episode using current hyperparameters takes relatively long. Rollout length, number of epochs maybe modified for better optimization
- The algorithm converges to 30 while there is more space to improve. A larger entropy coeffient may encourage more exploration for the agent to further improve. 
* Algorithms such as A3C, D4PG would be interesting to employ for this distributed training environment. 
