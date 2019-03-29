# SNN
An example of a generic stochastic gradient descent neural network.
Project in progress.

# Description

I've written something neural network-like before for my Zombie Dice game, but never a full, proper neural network program. So instead of just starting off with a simple model and expanding from there, I've decided to start in high gear and write a generic stochastic neural network. By generic, I mean that the number of layers and the number of neurons in each of those layers isn't set; you can decide what they will be before building the network. The advantage here is that if I want to change those parameters, then I can. The disadvantage is that I have to figure out the most efficient way to calculate all the partial derivatives for the back propogation on a network that could be of any size. I have to do this without making a single error because finding a bug in this later will be very difficult. I don't think it'll be too difficult to switch between stochastic and batch later, but that's a problem to solve later.

Here's an example of a 2-3-1 network.

                        N[1][0].bias
                        N[1][0].sum
                        N[1][0].sig
    N[0][0].bias        N[1][0].weight[0]
    N[0][0].sum         N[1][0].weight[1]
    N[0][0].sig                              N[2][0].bias
    N[0][0].weight[]    N[1][1].bias         N[2][0].sum
                        N[1][1].sum          N[2][0].sig          L()
    N[0][0].bias        N[1][1].sig          N[2][0].weight[0]
    N[0][0].sum         N[1][1].weight[0]    N[2][0].weight[1]
    N[0][0].sig         N[1][1].weight[1]    N[2][0].weight[2]
    N[0][0].weight[]
                        N[1][2].bias
                        N[1][2].sum
                        N[1][2].sig
                        N[1][2].weight[0]
                        N[1][2].weight[1]

A neuron in the first column is actually used as the input node. It makes the code simpler this way. A neuron will have as many weights as there are nodes in the previous layer. L() is the loss or error function.

To calculate the partial derivative of some given weight n[i][j].weight[k], then I need to calc

             d n[p][q].sum
          -------------------   where p = i, i+1, i+2, ... I
          d n[i][j].weight[k]         q = 0, 1, 2,     ... J

           and

           d n[p][q].sigmoid
          -------------------   where p = i, i+1, i+2, ... I
          d n[i][j].weight[k]         q = 0, 1, 2,     ... J

          for all weights. And obviously also

            d n[p][q].sum       where p = i, i+1, i+2, ... I
          ------------------          q = 0, 1, 2,     ... J
            d n[i][j].bias

           d n[p][q].sigmoid
          -------------------   where p = i, i+1, i+2, ... I
            d n[i][j].bias            q = 0, 1, 2,     ... J

           for all biases.

