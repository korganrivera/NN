/* an example stochastic gradient descent NN. Number of layers and depth of
 * each layer will be set at run time.
 * I'll be able to choose number of layers and number of nodes in each layer
 * later, but while I'm coding, I'll assume the network looks like this.

          N1
    X1
          N2    out
    X2
          N3

 * Fri Mar 15 16:59:21 CDT 2019
 *
 * Writing the network is the easy part. Preprocessing the data will be where
 * the real work is.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linear_algebra.h"

double sigmoid(double x);

// set some defaults. replace these with config file options later.
#define LAYERS 3
#define INPUT_ROWS 4

#define LR 0.1

typedef struct{
    double *weight;
    double bias;
    double output;
}neuron;

// points to each layer and each neuron. Layers may have different depths.
neuron **network;

int main(int argc, char **argv){
    unsigned i, j, k;
    unsigned node_count[] = {2, 3, 1, 4, 5, 3, 5, 2, 1}; // Gives me random node counts for each layer for testing with.

    // test data.
    int data[][3] = {
        { -2, -1, 1},
        { 25,  6, 0},
        { 17,  4, 0},
        {-15, -6, 1}
    };

    // malloc space for input, hidden, and output layers.
    network = malloc(LAYERS * sizeof(neuron*));

    // malloc space for nodes in each layer.
    for(i = 0; i < LAYERS; i++){
        network[i] = malloc(node_count[i] * sizeof(neuron));
    }

    // malloc space for weights in each node in each layer.
    for(i = 1; i < LAYERS; i++){
        for(j = 0; j < node_count[i]; j++){
            network[i][j].weight = malloc(node_count[i - 1] * sizeof(double));
        }
    }

    // initialise nodes.
    for(i = 1; i < LAYERS; i++){
        for(j = 0; j < node_count[i]; j++){
            network[i][j].bias = 0.0;
            network[i][j].output = 0.0;
            for(k = 0; k < node_count[i - 1]; k++){
                network[i][j].weight[k] = 1.0;
            }
        }
    }

    // load one data point into first layer.
    for(i = 0; i < node_count[i]; i++){
        network[0][i].output = data[0][i];
    }

    // feed forward.
    for(i = 1; i < LAYERS; i++){
        for(j = 0; j < node_count[i]; j++){
            for(k = 0; k < node_count[i - 1]; k++){
                network[i][j].output += network[i][j].weight[k] * network[i - 1][k].output;
            }
            network[i][j].output = sigmoid(network[i][j].output);
        }
    }

    // display output to check.
    for(j = 0; j < node_count[LAYERS - 2]; j++){
        printf("output from node %u: %lf\n", j, network[LAYERS - 1][j].output);
    }

    // next steps, calculate the MSE, then backprop.

    puts("beep.");
}




double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double deriv_sigmoid(double x){
    double fx = sigmoid(x);
    return fx * (1.0 - fx);
}

double tanh(double x){
    double a = exp(x);
    double b = exp(-x);
    return (a - b) / (a + b);
}

double deriv_tanh(double x){
    double a = tanh(x);
    return 1 - a * a;
}

double mse(double *true, double *pred, unsigned size){
    double error = 0.0;
    for(unsigned i = 0; i < size; i++){
        double diff = true[i] - pred[i];
        error += diff * diff;
    }
    return error / size;
}

