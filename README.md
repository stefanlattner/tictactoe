

# Fully Differentiable "Tik Tak Toe" aka "Noughts and Crosses", "Connect Four", ...

<div style="text-align: right"> Stefan Lattner </div>

This experiment shows that learning to play a game through self-play can be 
achieved through end-to-end optimization without classical reinforcement 
learning.

At a given state in the game, the next action is chosen by sampling from a 
Neural Network's (NN) prediction (using softmax over all valid actions). In 
order to preserve a gradient when sampling, a strategy similar to the 
 straight-through estimator [1] is employed. This is, the sampled 
 configuration is expressed by the corresponding probability distribution 
 *plus* a constant diff function, which is not part of the computation graph.
 
 The whole game is finally expressed as a computation graph. In case of a 
 defeat, the *last step* is penalized, in case of a success, the last step 
 is positively reinforced. This results in a weight update for the NN over 
 all steps of the game.
 
 After playing 500k games in self-play, the network achieves a success rate of 
 \>99% against a random-guessing opponent. The experiment shows that it 
 is possible to optimize agents taking series of discrete actions using 
 end-to-end gradient descent optimization.

### Prerequisites ###

* [PyTorch](http://www.pytorch.org)
* Numpy

**Install PyTorch** following [this link](http://www.pytorch.org).


[1] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. *Estimating or 
propagating gradients through
stochastic neurons for conditional computation.* arXiv preprint arXiv:1308
.3432, 2013.