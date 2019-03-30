/* an example stochastic gradient descent NN. Number of layers and depth of
 * each layer will be set at run time.
 * I'll be able to choose number of layers and number of nodes in each layer
 * later, but while I'm coding, I'll assume the network looks like this.

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
        }

        /* So, here's the deal with the partial derivatives.
         * (after about 6 hours of studying the problem...)
         * Given a weight or bias, I think I have an algorithm that will give
         * me the correct partial derivatives to calculate in the right order.
         * If the given weight is n[i][j].weight[k], then I need to calc
         *
         *    d n[p][q].sum
         * -------------------   where p = i, i+1, i+2, ... I
         * d n[i][j].weight[k]         q = 0, 1, 2,     ... J
         *
         *  and
         *
         *  d n[p][q].sigmoid
         * -------------------   where p = i, i+1, i+2, ... I
         * d n[i][j].weight[k]         q = 0, 1, 2,     ... J
         *
         * for all weights. And obviously also
         *
         *   d n[p][q].sum       where p = i, i+1, i+2, ... I
         * ------------------          q = 0, 1, 2,     ... J
         *   d n[i][j].bias
         *
         *  d n[p][q].sigmoid
         * -------------------   where p = i, i+1, i+2, ... I
         *   d n[i][j].bias            q = 0, 1, 2,     ... J
         *
         *  for all biases.
         *
         * When I include the sigmoid_deriv() function, then I have all the
         * parts I need to calculate all the partial derivatives for all the
         * weights and biases. Debugging this will be a fucking nightmare, so
         * get it right the first time!
         *
         *
         * Fri Mar 29 21:30:11 CDT 2019
         * Pretty excited! After about a million hours, I think I've come up
         * with the simplest way to calculate all those partial derivatives.
         * I'm going to add two fields to my neurons, and they are dsig_dsum
         * and dsum_dw. Starting at the layer that the weight in question is, I
         * calculate those values for each neuron in that layer. Then I repeat
         * for the layers ahead of it until I reach the last neurons. That
         * gives me dypred_dw. I use that to calculate dL_dypred, which is
         * easy.  And that's all there is too it! Isn't that insane!? The
         * entire nest of partial derivatives collapses to a super simple
         * algorithm!
         * */

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

