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
double sq_error(double *ytrue, double *ypred, unsigned size);

// set some defaults. replace these with config file options later.
#define LAYERS 3

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

    // malloc space for squared error.
    double *error = malloc(node_count[LAYERS - 1] * sizeof(double));

    // malloc space for sum error.
    double *sum_error = malloc(node_count[LAYERS - 1] * sizeof(double));

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

    // initialise squared error vector and sum_error.
    for(j = 0; j < node_count[LAYERS - 1]; j++){
        error[j] = 0.0;
        sum_error[j] = 0.0;
    }

    // process each data point.
    for(unsigned data_index = 0; data_index < 4; data_index++){

        puts("\nnew data:");
        // load one data point into first layer.
        for(i = 0; i < node_count[i]; i++){
            network[0][i].output = data[data_index][i];
        }

        // feed forward.
        for(i = 1; i < LAYERS; i++){
            for(j = 0; j < node_count[i]; j++){
                for(k = 0; k < node_count[i - 1]; k++){
                    network[i][j].output += network[i][j].weight[k] * network[i - 1][k].output;
                }
                //printf("neuron[%u][%u].output = %lf\n", i, j, network[i][j].output);
                printf("network[%u][%u].output = %lf\n", i, j, network[i][j].output);
                network[i][j].output = sigmoid(network[i][j].output);
                printf("sigmoid:network[%u][%u].output = %lf\n", i, j, network[i][j].output);
                //printf("sigmoid(neuron[%u][%u].output) = %lf\n", i, j, network[i][j].output);
            }
        }

        // next steps, calculate the MSE, then backprop.
        // make vectors of Ytrue and Ypred
        // send to mse, get error in return.
        // calculate every partial derivative in the network for error.
        // adjust all weights and biases.
        // choose next data sample and repeat.
        // repeat whole data set 1000 times or so, until error is wee.
        // use it for future predictions.
        for(j = 0; j < node_count[LAYERS - 1]; j++){
            error[j] = data[data_index][2 + j] - network[LAYERS - 1][j].output;
            error[j] *= error[j];
        }

        // add data's error to sum_error.
        for(j = 0; j < node_count[LAYERS - 1]; j++){
            sum_error[j] += error[j];
        }

        // show error.
        printf("error: ");
        for(j = 0; j < node_count[LAYERS - 1]; j++)
            printf("%lf, %lf \n", error[j], sum_error[j]);
    } // each data point processed.

    // finish calculating MSE.
    for(j = 0; j < node_count[LAYERS - 1]; j++){
        sum_error[j] /= 4;
    }

    // show error.
    printf("MSE: ");
    for(j = 0; j < node_count[LAYERS - 1]; j++)
        printf("%lf \n", sum_error[j]);

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

double sq_error(double *ytrue, double *ypred, unsigned size){
    double error = 0.0;
    for(unsigned i = 0; i < size; i++){
        double diff = ytrue[i] - ypred[i];
        error += diff * diff;
    }
}

