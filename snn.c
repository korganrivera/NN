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
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linear_algebra.h"

// set some defaults. replace these with config file options later.
#define INPUT_ROWS 4
#define INPUT_COLS 2
#define H_LAYERS 1
#define L1_NODES 3
#define OUTPUTS 1

typedef struct{
    double *weight;
    double bias;
    double output;
}neuron;

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

void feedforward(double *x, neuron *node){
    node->output = sigmoid(dot(INPUT_ROWS, x, node->weight) + node->bias);
}

// points to each layer and each neuron. Layers may have different depths.
neuron **network;

// points to dataset.
double **input;

// learning rate.
double learn_rate;

int main(int argc, char **argv){
    unsigned i, j;

    // malloc space for input.
    input = malloc(INPUT_ROWS * sizeof(double*));
    for(i = 0; i < INPUT_ROWS; i++){
        input[i] = malloc(INPUT_COLS * sizeof(double));
    }

    // malloc space for hidden layers and output layer.
    network = malloc((H_LAYERS + 1) * sizeof(neuron*));

    // malloc space for nodes in first layer.
    network[0] = malloc(L1_NODES * sizeof(neuron));

    // malloc space for output nodes.
    network[1] = malloc(OUTPUTS * sizeof(neuron));

    // malloc space for weights in each node of first layer.
    for(i = 0; i < L1_NODES; i++){
        network[0][i].weight = malloc(INPUT_ROWS * sizeof(double));
    }

    // malloc space for weights in each node of output layer.
    for(i = 0; i < OUTPUTS; i++){
        network[1][i].weight = malloc(L1_NODES * sizeof(double));
    }

    // initialise weights array and biases in each node of first layer.
    for(i = 0; i < L1_NODES; i++){
        for(j = 0; j < INPUT_ROWS; j++){
            network[0][i].weight[j] = 1.0;
            network[0][i].bias      = 0.0;
        }
    }

    // initialise weights array and biases in each node of output layer.
    for(i = 0; i < OUTPUTS; i++){
        for(j = 0; j < L1_NODES; j++){
            network[1][i].weight[j] = 1.0;
            network[1][i].bias      = 0.0;
        }
    }

    // test data set.
    input[0] = (double[2]) {-2,  -1};
    input[1] = (double[2]) {25,   6};
    input[2] = (double[2]) {17,   4};
    input[3] = (double[2]) {-15, -6};

    // push data through the network.
    for(j = 0; j < 4; j++){
        // push data through first layer.
        for(i = 0; i < L1_NODES; i++){
            feedforward(input[j], &network[0][i]);
        }

        // make an array of the outputs from the first layer.
        double *layer_out = malloc(L1_NODES * sizeof(double));

        // copy outputs from first layer into layer_out.
        for(i = 0; i < L1_NODES; i++){
            layer_out[i] = network[0][i].output;
        }

        // feedforward first layer to output layer.
        for(i = 0; i < OUTPUTS; i++){
            feedforward(layer_out, &network[1][i]);
        }

        // now output neuron should contain its output value.
        printf("input: %7.3lf %7.3lf | output %7.3lf\n", input[j][0], input[j][1], network[1][0]);
    }

    // next step: back prop.

    puts("beep.");
}




