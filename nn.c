
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LR 0.1
#define EPOCHS 1000

typedef struct{
    double b;
    double *w;
    double sum;
    double sig;

    double dsig_dsum;
    double dsum_dw;
    double dsig_dw;
    double dsum_db;
    double dsig_db;

    double *w_adjustment;
    double b_adjustment;
}neuron;

double sigmoid(double x);
double deriv_sigmoid(double x);
void feed_forward(neuron **n, unsigned *nodes);

int main(int argc, char **argv){
    neuron **network;
    unsigned i, j, k;
    unsigned node_count[] = {2, 2, 1};

    // test data.
    int data[][3] = {
        { -2, -1, 1},
        { 25,  6, 0},
        { 17,  4, 0},
        {-15, -6, 1}
    };

    printf("creating network...");
    // malloc space for input, hidden, and output layers.
    network = malloc(3 * sizeof(neuron*));

    // malloc space for nodes in each layer.
    for(i = 0; i < 3; i++){
        network[i] = malloc(node_count[i] * sizeof(neuron));
    }

    // malloc space for weights and there adjustments in each node in each layer.
    for(i = 1; i < 3; i++){
        for(j = 0; j < node_count[i]; j++){
            network[i][j].w = malloc(node_count[i - 1] * sizeof(double));
            network[i][j].w_adjustment = malloc(node_count[i - 1] * sizeof(double));
        }
    }
    printf("done.\ninitialising...");
    // initialise weights and biases.
    for(i = 1; i < 3; i++){
        for(j = 0; j < node_count[i]; j++){
            for(k = 0; k < node_count[i - 1]; k++){
                network[i][j].w[k] = 1.0;
            }
            network[i][j].b = 0.0;
        }
    }

    printf("done\ntraining...");
    for(unsigned e = 0; e < EPOCHS; e++){
        for(unsigned d = 0; d < 4; d++){
            // load data.
            for(j = 0; j < node_count[j]; j++)
                network[0][j].sig = data[d][j];

            feed_forward(network, node_count);

            // calculate error and dE/dsig.
            double diff    = data[d][2] - network[2][0].sig;
            double error   = diff * diff;
            double dEdsig  = -2.0 * diff;

            // back prop the stupid way.
            // w100.------------------------------------------------------------
            network[1][0].dsum_dw = network[0][0].sig;
            network[1][0].dsig_dw = network[1][0].dsig_dsum * network[1][0].dsum_dw;

            network[1][1].dsum_dw = 0;
            network[1][1].dsig_dw = network[1][1].dsig_dsum * network[1][1].dsum_dw;

            network[2][0].dsum_dw = network[2][0].w[0] * network[1][0].dsig_dw + network[2][0].w[1] * network[1][1].dsig_dw;
            network[2][0].dsig_dw = network[2][0].dsig_dsum * network[2][0].dsum_dw;

            double dEdw100 = dEdsig * network[2][0].dsig_dw;

            // set w100's adjustment.
            network[1][0].w_adjustment[0] = LR * dEdw100;

            // w101.------------------------------------------------------------
            network[1][0].dsum_dw = network[0][1].sig;
            network[1][0].dsig_dw = network[1][0].dsig_dsum * network[1][0].dsum_dw;

            network[1][1].dsum_dw = 0;
            network[1][1].dsig_dw = network[1][1].dsig_dsum * network[1][1].dsum_dw;

            network[2][0].dsum_dw = network[2][0].w[0] * network[1][0].dsig_dw + network[2][0].w[1] * network[1][1].dsig_dw;
            network[2][0].dsig_dw = network[2][0].dsig_dsum * network[2][0].dsum_dw;

            double dEdw101 = dEdsig * network[2][0].dsig_dw;

            // set w101's adjustment.
            network[1][0].w_adjustment[1] = LR * dEdw101;

            // w110.------------------------------------------------------------
            network[1][0].dsum_dw = 0;
            network[1][0].dsig_dw = network[1][0].dsig_dsum * network[1][0].dsum_dw;

            network[1][1].dsum_dw = network[0][0].sig;
            network[1][1].dsig_dw = network[1][1].dsig_dsum * network[1][1].dsum_dw;

            network[2][0].dsum_dw = network[2][0].w[0] * network[1][0].dsig_dw + network[2][0].w[1] * network[1][1].dsig_dw;
            network[2][0].dsig_dw = network[2][0].dsig_dsum * network[2][0].dsum_dw;

            double dEdw110 = dEdsig * network[2][0].dsig_dw;

            // set w100's adjustment.
            network[1][1].w_adjustment[0] = LR * dEdw110;

            // w111.------------------------------------------------------------
            network[1][0].dsum_dw = 0;
            network[1][0].dsig_dw = network[1][0].dsig_dsum * network[1][0].dsum_dw;

            network[1][1].dsum_dw = network[0][1].sig;
            network[1][1].dsig_dw = network[1][1].dsig_dsum * network[1][1].dsum_dw;

            network[2][0].dsum_dw = network[2][0].w[0] * network[1][0].dsig_dw + network[2][0].w[1] * network[1][1].dsig_dw;
            network[2][0].dsig_dw = network[2][0].dsig_dsum * network[2][0].dsum_dw;

            double dEdw111 = dEdsig * network[2][0].dsig_dw;

            // set w111's adjustment.
            network[1][1].w_adjustment[1] = LR * dEdw111;

            // w200.------------------------------------------------------------
            network[1][0].dsum_dw = 0;
            network[1][0].dsig_dw = network[1][0].dsig_dsum * network[1][0].dsum_dw;

            network[1][1].dsum_dw = 0;
            network[1][1].dsig_dw = network[1][1].dsig_dsum * network[1][1].dsum_dw;

            network[2][0].dsum_dw = network[1][0].sig;
            network[2][0].dsig_dw = network[2][0].dsig_dsum * network[2][0].dsum_dw;

            double dEdw200 = dEdsig * network[2][0].dsig_dw;

            // set w200's adjustment.
            network[2][0].w_adjustment[0] = LR * dEdw200;

            // w201.------------------------------------------------------------
            network[1][0].dsum_dw = 0;
            network[1][0].dsig_dw = network[1][0].dsig_dsum * network[1][0].dsum_dw;

            network[1][1].dsum_dw = 0;
            network[1][1].dsig_dw = network[1][1].dsig_dsum * network[1][1].dsum_dw;

            network[2][0].dsum_dw = network[1][1].sig;
            network[2][0].dsig_dw = network[2][0].dsig_dsum * network[2][0].dsum_dw;

            double dEdw201 = dEdsig * network[2][0].dsig_dw;

            // set w201's adjustment.
            network[2][0].w_adjustment[1] = LR * dEdw201;

            // back propagate for the biases.
            for(i = 1; i < 3; i++){
                for(j = 0; j < node_count[i]; j++){

                    // calc dsum_db and dsig_db.
                    for(unsigned a = i; a < 3; a++){
                        for(unsigned b = 0; b < node_count[a]; b++){
                            if(a > i){
                                // network[a][b].dsum_db = network[a][b].w[0] * network[a - 1][0].dsig_db + network[a][b].w[1] * network[a - 1][1].dsig_db;
                                network[a][b].dsum_db = 0;
                                for(unsigned c = 0; c < node_count[a - 1]; c++){
                                    network[a][b].dsum_db += network[a][b].w[c] * network[a - 1][c].dsig_db;
                                }
                            }

                            else if(a == i && b == j)
                                network[a][b].dsum_db = 1;
                            else
                                network[a][b].dsum_db = 0;

                            network[a][b].dsig_db = network[a][b].dsig_dsum * network[a][b].dsum_db;
                        }
                    }

                    // calc bias adjustment.
                    network[i][j].b_adjustment = LR * dEdsig * network[2][0].dsig_db;
                }
            }

            // do same for biases
            // b10.------------------------------------------------------------
//            network[1][0].dsum_db = 1;
//            network[1][0].dsig_db = network[1][0].dsig_dsum * network[1][0].dsum_db;
//
//            network[1][1].dsum_db = 0;
//            network[1][1].dsig_db = network[1][1].dsig_dsum * network[1][1].dsum_db;
//
//            network[2][0].dsum_db = network[2][0].w[0] * network[1][0].dsig_db + network[2][0].w[1] * network[1][1].dsig_db;
//            network[2][0].dsig_db = network[2][0].dsig_dsum * network[2][0].dsum_db;
//
//            double dEdb10 = dEdsig * network[2][0].dsig_db;
//
//            // set b10's adjustment.
//            network[1][0].b_adjustment = LR * dEdb10;
//
//            // b11.------------------------------------------------------------
//            network[1][0].dsum_db = 0;
//            network[1][0].dsig_db = network[1][0].dsig_dsum * network[1][0].dsum_db;
//
//            network[1][1].dsum_db = 1;
//            network[1][1].dsig_db = network[1][1].dsig_dsum * network[1][1].dsum_db;
//
//            network[2][0].dsum_db = network[2][0].w[0] * network[1][0].dsig_db + network[2][0].w[1] * network[1][1].dsig_db;
//            network[2][0].dsig_db = network[2][0].dsig_dsum * network[2][0].dsum_db;
//
//            double dEdb11 = dEdsig * network[2][0].dsig_db;
//
//            // set b11's adjustment.
//            network[1][1].b_adjustment = LR * dEdb11;
//
//            // b20.------------------------------------------------------------
//            network[1][0].dsum_db = 0;
//            network[1][0].dsig_db = network[1][0].dsig_dsum * network[1][0].dsum_db;
//
//            network[1][1].dsum_db = 0;
//            network[1][1].dsig_db = network[1][1].dsig_dsum * network[1][1].dsum_db;
//
//            network[2][0].dsum_db = 1;
//            network[2][0].dsig_db = network[2][0].dsig_dsum * network[2][0].dsum_db;
//
//            double dEdb20 = dEdsig * network[2][0].dsig_db;
//
//            // set b20's adjustment.
//            network[2][0].b_adjustment = LR * dEdb20;

            // adjust all the weights.
            for(i = 1; i < 3; i++)
                for(j = 0; j < node_count[i]; j++)
                    for(k = 0; k < node_count[i - 1]; k++)
                        network[i][j].w[k] -= network[i][j].w_adjustment[k];

            // adjust all the biases.
            for(i = 1; i < 3; i++)
                for(j = 0; j < node_count[i]; j++)
                    network[i][j].b -= network[i][j].b_adjustment;
        }
    }
    puts("done.\nTesting:\n");
    // try it out.
    // Emily
    int weight, height;
    weight = 128;
    height = 63;
    network[0][0].sig = weight - 135;
    network[0][1].sig = height - 66;
    feed_forward(network, node_count);
    printf("Emily:\nweight: %u lb\nheight: %u \"\npredicted gender: ", weight, height);
    if(network[2][0].sig >= 0.5)
        puts("female");
    else
        puts("male");
    printf("(score = %lf)\n\n", network[2][0].sig);

    // Frank
    weight = 155;
    height = 68;
    network[0][0].sig = weight - 135;
    network[0][1].sig = height - 66;
    feed_forward(network, node_count);
    printf("Frank:\nweight: %u lb\nheight: %u \"\npredicted gender: ", weight, height);
    if(network[2][0].sig >= 0.5)
        puts("female");
    else
        puts("male");
    printf("(score = %lf)\n", network[2][0].sig);
}

void feed_forward(neuron **n, unsigned *nodes){
unsigned i, j, k;
    for(i = 1; i < 3; i++){
        for(j = 0; j < nodes[i]; j++){
            n[i][j].sum = 0;
            for(k = 0; k < nodes[i - 1]; k++){
                n[i][j].sum += n[i][j].w[k] * n[i - 1][k].sig;
            }
            n[i][j].sum += n[i][j].b;
            n[i][j].sig = sigmoid(n[i][j].sum);

            // calc dsig_dsum while we're here even though it's not really part of feed forwarding.
            n[i][j].dsig_dsum = n[i][j].sig * (1 - n[i][j].sig);
        }
    }
}

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}
