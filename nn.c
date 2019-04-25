
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LR 0.1
#define EPOCHS 2000
#define LAYERS 4

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
    unsigned node_count[] = {2, 2, 3, 1};

    // test data.
    int data[][3] = {
        { -2, -1, 1},
        { 25,  6, 0},
        { 17,  4, 0},
        {-15, -6, 1}
    };

    // check size of node_count[] in case I set LAYERS too high. This is for debugging.
    unsigned size_nc = sizeof(node_count)/sizeof(unsigned);
    if(size_nc < LAYERS){
        puts("node_count is too short brah.");
        exit(1);
    }

    printf("NEURAL NETWORK\n  layers=%u\n", LAYERS);
    for(i = 0; i < LAYERS; i++)
        printf("    layer %u: %u neurons\n", i, node_count[i]);
    putchar('\n');

    printf("creating network...");
    // malloc space for input, hidden, and output layers.
    if((network = malloc(LAYERS * sizeof(neuron*))) == NULL) { puts("malloc failed"); exit(1); }

    // malloc space for nodes in each layer.
    for(i = 0; i < LAYERS; i++){
        if((network[i] = malloc(node_count[i] * sizeof(neuron))) == NULL) { puts("malloc failed"); exit(1); }
    }

    // malloc space for weights and their adjustments in each node in each layer.
    for(i = 1; i < LAYERS; i++){
        for(j = 0; j < node_count[i]; j++){
            if((network[i][j].w = malloc(node_count[i - 1] * sizeof(double))) == NULL) { puts("malloc failed"); exit(1); }
            if((network[i][j].w_adjustment = malloc(node_count[i - 1] * sizeof(double))) == NULL) { puts("malloc failed"); exit(1); }
        }
    }
    printf("done.\ninitialising...");
    // initialise weights and biases.
    for(i = 1; i < LAYERS; i++){
        for(j = 0; j < node_count[i]; j++){
            for(k = 0; k < node_count[i - 1]; k++){
                network[i][j].w[k] = 1.0;
            }
            network[i][j].b = 0.0;
        }
    }

    puts("done\ntraining...");
    for(unsigned e = 0; e < EPOCHS; e++){
        for(unsigned d = 0; d < 4; d++){
            // load data.
            for(j = 0; j < node_count[0]; j++)
                network[0][j].sig = data[d][j];

            // feed forward.
            feed_forward(network, node_count);

            // calculate error and dE/dsig.
            double diff    = data[d][2] - network[LAYERS - 1][0].sig;
            double error   = diff * diff;

            // if I've run at least 1000 epochs and the error is < 2%, bail.
            if(error < 0.02 && e > 1000){
                printf("error is low enough. Ending training...");
                goto beep;
            }

            double dEdsig  = -2.0 * diff;
            if(e % 100 == 0) printf("error = %lf\n", error);
            // back propagate for the weights.
            for(i = 1; i < LAYERS; i++){
                for(j = 0; j < node_count[i]; j++){
                    for(k = 0; k < node_count[i - 1]; k++){

                        // calc dsum_dw and dsig_dw.
                        for(unsigned a = i; a < LAYERS; a++){
                            for(unsigned b = 0; b < node_count[a]; b++){

                                if(a > i){
                                    network[a][b].dsum_dw = 0;
                                    for(unsigned c = 0; c < node_count[a - 1]; c++){
                                        network[a][b].dsum_dw += network[a][b].w[c] * network[a - 1][c].dsig_dw;
                                    }
                                }
                                else if(a == i && b == j)
                                    network[a][b].dsum_dw = network[a - 1][k].sig;
                                else
                                    network[a][b].dsum_dw = 0;

                                network[a][b].dsig_dw = network[a][b].dsig_dsum * network[LAYERS - 1][0].dsum_dw;
                            }
                        }

                        // calc weight adjustment.
                        network[i][j].w_adjustment[k] = LR * dEdsig * network[LAYERS - 1][0].dsig_dw;
                    }
                }
            }

            // back propagate for the biases.
            for(i = 1; i < LAYERS; i++){
                for(j = 0; j < node_count[i]; j++){

                    // calc dsum_db and dsig_db.
                    for(unsigned a = i; a < LAYERS; a++){
                        for(unsigned b = 0; b < node_count[a]; b++){
                            if(a > i){
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
                    network[i][j].b_adjustment = LR * dEdsig * network[LAYERS - 1][0].dsig_db;
                }
            }

            // adjust all the weights.
            for(i = 1; i < LAYERS; i++)
                for(j = 0; j < node_count[i]; j++)
                    for(k = 0; k < node_count[i - 1]; k++)
                        network[i][j].w[k] -= network[i][j].w_adjustment[k];

            // adjust all the biases.
            for(i = 1; i < LAYERS; i++)
                for(j = 0; j < node_count[i]; j++)
                    network[i][j].b -= network[i][j].b_adjustment;
        }
    }

    beep:
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
    puts((network[LAYERS - 1][0].sig >= 0.5) ? "female" : "male");
    printf("(score = %lf)\n", network[LAYERS - 1][0].sig);

    // Frank
    weight = 155;
    height = 68;
    network[0][0].sig = weight - 135;
    network[0][1].sig = height - 66;
    feed_forward(network, node_count);
    printf("Frank:\nweight: %u lb\nheight: %u \"\npredicted gender: ", weight, height);
    puts((network[LAYERS - 1][0].sig >= 0.5) ? "female" : "male");
    printf("(score = %lf)\n", network[LAYERS - 1][0].sig);
}

void feed_forward(neuron **n, unsigned *nodes){
unsigned i, j, k;

    for(i = 1; i < LAYERS; i++){
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
