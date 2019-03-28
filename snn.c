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
double deriv_sigmoid(double x);

// set some defaults. replace these with config file options later.
#define LAYERS 3

#define LR 0.1

typedef struct{
    double *weight;
    double bias;
    double sum;
    double sigmoid;
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

    // malloc space for y_pred.
    double *y_pred = malloc(node_count[LAYERS - 1] * sizeof(double));

    // initialise nodes.
    for(i = 1; i < LAYERS; i++){
        for(j = 0; j < node_count[i]; j++){
            network[i][j].bias = 0.0;
            network[i][j].sum = 0.0;
            network[i][j].sigmoid = 0.0;
            for(k = 0; k < node_count[i - 1]; k++){
                network[i][j].weight[k] = 1.0;
            }
        }
    }

    // initialise squared error vector.
    for(j = 0; j < node_count[LAYERS - 1]; j++){
        error[j] = 0.0;
    }

    // process each data point.
    for(unsigned data_index = 0; data_index < 4; data_index++){

        puts("\nnew data:");
        // load one data point into first layer.
        for(i = 0; i < node_count[i]; i++){
            network[0][i].sigmoid = data[data_index][i];
        }

        // feed forward.
        for(i = 1; i < LAYERS; i++){
            for(j = 0; j < node_count[i]; j++){
                for(k = 0; k < node_count[i - 1]; k++){
                    network[i][j].sum += network[i][j].weight[k] * network[i - 1][k].sigmoid;
                }
                printf("network[%u][%u].sum = %lf\n", i, j, network[i][j].sum);
                network[i][j].sigmoid = sigmoid(network[i][j].sum);
                printf("network[%u][%u].sigmoid = %lf\n", i, j, network[i][j].sigmoid);
            }
        }

        // backprop: calc all partial derivatives for each output.
        // This is spaghetti.


        double *dL_dypredw = malloc(node_count[LAYERS - 2] * sizeof(double));
        double **dh_dw = malloc(node_count[LAYERS - 2] * sizeof(double));
        for(i = 0; i < node_count[LAYERS - 2]; i++){
            dh_dw[i] = malloc(node_count[LAYERS - 3] * sizeof(double));
        }


        // put outputs into y_pred. calc error between true and pred.
        for(j = 0; j < node_count[LAYERS - 1]; j++){
            y_pred[j] = network[LAYERS - 1][j].sigmoid;
            error[j] = data[data_index][2 + j] - y_pred[j];
            error[j] *= error[j];
            printf("sq_error: %lf\n", error[j]);

            // dL / d(y_pred).
            dL_dypredw[j] = -2 * error[j];

            // d_ypred / d_w, where w is weights attached to this neuron in last layer.
            for(i = 0; i < node_count[LAYERS - 2]; i++){
                dL_dypredw[i] = network[LAYERS - 2][i].sigmoid * deriv_sigmoid(network[LAYERS -1][j].sum);
            }

            // d_ypred / d_bias
            double dypred_d_bias = deriv_sigmoid(network[LAYERS -1][j].sum);

            

        }


        // d(y_pred) / dw[], where w[] is all the weights attached to that neuron.
        double dypred_dw[node_count[LAYERS - 1]][node_count[LAYERS - 2]];


        // calc partial derivatives. dL/dYpred = -2 * (y_true - y_pred)
        for(j = 0; j < node_count[LAYERS - 1]; j++){
           // dL_dypred[j] = -2 * error[j];
        }

        // dh[] / dw[].
        for(i = 0; i < node_count[LAYERS - 2]; i++){
            for(j = 0; j < node_count[LAYERS - 3]; j++){
                dh_dw[i][j] = network[LAYERS - 3][j].sigmoid * deriv_sigmoid( network[LAYERS - 2][i].sum);
            }
        }

        // This is getting ridiculous. I'm going to finish this when I've drawn it all out.


    } // each data point processed.

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

