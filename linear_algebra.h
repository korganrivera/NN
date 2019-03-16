#include <assert.h>
#include <limits.h>

int echo_array(unsigned rows, unsigned cols, double **array);

int echo_vector(unsigned rows, double *array);

int copy(unsigned rows, unsigned cols, double **A, double **B);

int add_v(unsigned rows, double *vector1, double *vector2, double *vector3);

int scale_x(unsigned rows, unsigned cols, double factor, double **array);

int scale_v(unsigned rows, double factor, double *vector);

int transpose(unsigned rows, unsigned cols, double **array, double **trans);

int multiply(unsigned a_rows, unsigned a_cols, unsigned b_rows, unsigned b_cols, double **A, double **B, double **C);

int multiply_x_by_v(unsigned rows, unsigned cols, double **array, double *vector1, double *vector2);

double determinant(unsigned rows, double **array);

int invert(unsigned rows, double **array);

double dot(unsigned rows, double *vector1, double *vector2);
