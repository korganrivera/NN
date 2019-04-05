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

#define DATA_WIDTH 3

typedef struct{
    double *weight;
    double bias;
    double sum;
    double sigmoid;
    double dsig_dsum;
    double dsum_dw;
    double dsig_dw;

    double dsum_db;
    double dsig_db;

    double *w_adjustment;
    double b_adjustment;
}neuron;

typedef struct{
    double diff;
    double sq_err;
    double de_dsig;
}error;



// points to each layer and each neuron. Layers may have different depths.
neuron **network;

int main(int argc, char **argv){
    unsigned i, j, k;
    unsigned node_count[] = {2, 2, 2, 4, 5, 3, 5, 2, 1}; // Gives me random node counts for each layer for testing with.

    // test data.
    int data[][DATA_WIDTH] = {
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

    // malloc space for weights and there adjustments in each node in each layer.
    for(i = 1; i < LAYERS; i++){
        for(j = 0; j < node_count[i]; j++){
            network[i][j].weight = malloc(node_count[i - 1] * sizeof(double));
            network[i][j].w_adjustment = malloc(node_count[i - 1] * sizeof(double));
        }
    }

    // malloc space for y_pred.
    double *error_sq_vector = malloc(node_count[LAYERS - 1] * sizeof(double));

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

    // malloc space for error vector.
    error *error_vector = malloc(node_count[LAYERS - 1] * sizeof(error));

    // initialise error vectors.
    for(j = 0; j < node_count[LAYERS - 1]; j++){
        error_vector[j].diff    = 0.0;
        error_vector[j].sq_err  = 0.0;
        error_vector[j].de_dsig = 0.0;
    }

for(unsigned epoch = 0; epoch < 100; epoch++){
    printf("----------epoch %u\n", epoch);
    // process each data point.
    for(unsigned data_index = 0; data_index < DATA_WIDTH; data_index++){

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
                network[i][j].sum += network[i][j].bias;
                network[i][j].sigmoid = sigmoid(network[i][j].sum);
            }
        }
//***************do this for every weigh and bias*************************
        // calculate partial derivatives for back propagation.

        for(unsigned i_w = 1; i_w < LAYERS; i_w++){
        //for(unsigned i_w = 1; i_w < 2; i_w++){
        for(unsigned j_w = 0; j_w < node_count[i_w]; j_w++){
        //for(unsigned j_w = 0; j_w < 1; j_w++){




        for(unsigned k_w = 0; k_w < node_count[i_w - 1]; k_w++){
        //for(unsigned k_w = 0; k_w < 1; k_w++){
        printf("\ni_w = %u, j_w=%u, j_w=%u\n", i_w, j_w, k_w);
        // the weight I want to process is network[i_w][j_w].weight[k_w].

                for(i = i_w; i < LAYERS; i++){
                    for(j = 0; j < node_count[i]; j++){

                        // calculate sum of node.
                        network[i][j].sum = 0;
                        for(k = 0; k < node_count[i - 1]; k++){
                            network[i][j].sum += network[i][j].weight[k] * network[i - 1][k].sigmoid;
                        }
                        network[i][j].sum += network[i][j].bias;

                        // calculate sigmoid of node.
                        network[i][j].sigmoid = sigmoid(network[i][j].sum);

                        // calcalate dsig_dsum of node.
                        network[i][j].dsig_dsum = deriv_sigmoid(network[i][j].sum);

                        // calculate dsum_dw for node.
                        if(i == i_w){
                            if(j == j_w){
                                network[i][j].dsum_dw = network[i - 1][j].sigmoid;
                            }
                            else{
                                network[i][j].dsum_dw = 0;
                            }
                        }
                        else{
                           network[i][j].dsum_dw = 0;
                           for(k = 0; k < node_count[i - 1]; k++){
                               network[i][j].dsum_dw += network[i][j].weight[k] * network[i - 1][k].dsig_dw;
                           }
                        }

                        // calculate dsig_dw for node.
                        network[i][j].dsig_dw = network[i][j].dsig_dsum * network[i][j].dsum_dw;


                        // calculate dsum_db for node. only calc once.
                        if(i_w == 0){

                            if(i == i_w){
                                if(j == j_w){
                                    network[i][j].dsum_db = 1;
                                }
                                else{
                                    network[i][j].dsum_db = 0;
                                }
                            }
                            else{
                               network[i][j].dsum_db = 0;
                               for(k = 0; k < node_count[i - 1]; k++){
                                   network[i][j].dsum_db += network[i][j].weight[k] * network[i - 1][k].dsig_db;
                               }
                            }

                            // calculate dsig_db for node.
                            network[i][j].dsig_db = network[i][j].dsig_dsum * network[i][j].dsum_db;
                        }





                        // print to check.
                        printf("network[%u][%u].sum = %lf\n", i, j, network[i][j].sum);
                        printf("network[%u][%u].sigmoid = %lf\n", i, j, network[i][j].sigmoid);
                        printf("network[%u][%u].dsig_dsum = %lf\n", i, j, network[i][j].dsig_dsum);
                        printf("network[%u][%u].dsum_dw = %lf\n", i, j, network[i][j].dsum_dw);
                        printf("network[%u][%u].dsig_dw = %lf\n", i, j, network[i][j].dsig_dw);
                    }
                }

                // calc diff, squared error, total error.
                double total_error = 0.0;
                for(j = 0; j < node_count[LAYERS - 1]; j++){
                    error_vector[j].diff = data[data_index][node_count[0] + j] - network[LAYERS - 1][j].sigmoid;
                    error_vector[j].sq_err = error_vector[j].diff * error_vector[j].diff;
                    printf("sq_error: %lf\n", error_vector[j].sq_err);
                    error_vector[j].de_dsig = -2.0 * error_vector[j].diff;
                    total_error += error_vector[j].sq_err;
                }
                printf("####################total_error: %lf\n", total_error);


                // calculate dL/dw.
                double dLdw = 0.0;
                for(j = 0; j < node_count[LAYERS - 1]; j++){
                    dLdw += error_vector[j].de_dsig * network[LAYERS -1][j].dsig_dw;
                    printf("de_dsig:%lf, dsig_dw: %lf\n", error_vector[j].de_dsig, network[LAYERS -1][j].dsig_dw);
                }

                printf("dLdw for w[%u][%u][%u]: %lf\n", i_w, j_w, k_w, dLdw);

                // adjust weight accordingly.
                printf("setting adjustment %lf for  w[%u][%u][%u].\n", LR * dLdw, i_w, j_w, k_w);
                network[i_w][j_w].w_adjustment[k_w] = LR * dLdw; // subtract this later.

        }
                // calculate dL/db.
                double dLdb = 0.0;
                for(j = 0; j < node_count[LAYERS - 1]; j++){
                    dLdb += error_vector[j].de_dsig * network[LAYERS -1][j].dsig_db;
                    printf("de_dsig:%lf, dsig_db: %lf\n", error_vector[j].de_dsig, network[LAYERS -1][j].dsig_db);
                }
                printf("dLdb for b[%u][%u]: %lf\n", i_w, j_w, dLdb);

                // adjust bias accordingly.
                printf("setting adjustment %lf for  b[%u][%u].\n", LR * dLdb, i_w, j_w);
                network[i_w][j_w].b_adjustment = LR * dLdb; // subtract this later.
        }
        }

        // adjust all the weights using the adjustment.
        for(i = 1; i < LAYERS; i++){
            for(j = 0; j < node_count[i]; j++){
                for(k = 0; k < node_count[i - 1]; k++){
                    network[i][j].weight[k] -= network[i][j].w_adjustment[k];
                    printf("w[%u][%u][%u] is now: %lf\n", i, j, k, network[i][j].weight[k]);
                }
                // adjust the biases too.
                network[i][j].bias -= network[i][j].b_adjustment;
                printf("b[%u][%u] is now: %lf\n", i, j, network[i][j].bias);
            }
        }

//****************************************
    } // each data point processed.
 }
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

