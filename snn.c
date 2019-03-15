/* an example stochastic gradient descent NN. Number of layers and depth of
 * each layer will be set at run time.
 * Fri Mar 15 16:59:21 CDT 2019
 */

#include <stdio.h>
#include <math.h>

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double deriv_sigmoid(double x){
    double fx = sigmoid(x);
    return fx * (1.0 - fx);
}

double mse(double *true, double *pred, unsigned size){
    double error = 0.0;

    for(unsigned i = 0; i < size; i++)
        error += (true[i] - pred[i]) * (true[i] - pred[i]);
    return error / size;
}

typedef struct{
    double *weights;
    double bias;
}neuron;

// points to each layer and each neuron. Layers may have different depths.
neuron **layer_node;

// points to dataset.
double **data;

// learning rate.
double learn_rate;

int main(int argc, char **argv){
    puts("Beep.");
}
