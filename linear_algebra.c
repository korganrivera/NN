/* Thu Feb 28 10:04:45 CST 2019
 * cleaning up my code from 4 years ago.  Contains basic linear algebra
 * functions that I use for linear regression.
 */

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "linear_algebra.h"

/* displays matrix.
 * usage example: echo_array(3, 4, array);
 */
int echo_array(unsigned rows, unsigned cols, double **array){
    if(!rows || !cols)
        return -1;
    // abort if pointer is NULL.
    assert(array != NULL && "echo_array(): pointer is NULL. Why?");
    // abort if unsigned wraparound could happen.
    assert(cols < (UINT_MAX / rows) && "echo_array(): unsigned wrap happened.");
    for(unsigned i = 0; i < rows; i++){
        for(unsigned j = 0; j < cols; j++){
            printf("%.3lf ", array[i][j]);
        }
        putchar('\n');
    }
    putchar('\n');
    return 1;
}


/* displays vector.
 * usage example: echo_vector(9, vector);
 */
int echo_vector(unsigned rows, double *array){
    if(!rows)
        return -1;
    assert(array != NULL && "echo_vector(): vector pointer is NULL. Why?");
    for(unsigned i = 0; i < rows; i++)
        printf("%.3lf ", array[i]);
    puts("\n");
    return 1;
}


/* copy first array over second.
 * Assumes both arrays have the given dimensions.
 * usage example: copy(3, 5, A, B);
 */
int copy(unsigned rows, unsigned cols, double **A, double **B){
    assert(A != NULL && B != NULL && "copy(): pointer is NULL. Why?");

    for(unsigned i = 0; i < rows; i++){
        for(unsigned j = 0; j < cols; j++)
            B[i][j] = A[i][j];
    }
}


// add two vectors together.
int add_v(unsigned rows, double *vector1, double *vector2, double *vector3){
    assert(vector1 != NULL && vector2 != NULL && vector3 != NULL && "add_v(): pointer is NULL. Why?");
    if(rows == 0)
        return -1;
    for(unsigned i = 0; i < rows; i++){
        vector3[i] = vector1[i] + vector2[i];

    return 1;
    }
}


/* scale a matrix by some factor.
 * usage example: scale(3, 5, 7.0, array);
 */
int scale_x(unsigned rows, unsigned cols, double factor, double **array){
    if(!rows || !cols)
        return -1;
    assert(array != NULL && "scale_x(): pointer is NULL. Why?");
    for(unsigned i = 0; i < rows; i++){
        for(unsigned j = 0; j < rows; j++){
            array[i][j] *= factor;
        }
    }
    return 1;
}


int scale_v(unsigned rows, double factor, double *vector){
    if(!rows)
        return -1;
    assert(vector != NULL && "scale_v(): pointer is NULL. Why?");
    for(unsigned i = 0; i < rows; i++)
        vector[i] *= factor;
}


/* transposes array into trans.
 * usage example: transpose(3, 5, x, x_trans);
 */
int transpose(unsigned rows, unsigned cols, double **array, double **trans){
    if(!rows || !cols) return -1;
    assert(array != NULL && trans != NULL && "transpose(): pointer is NULL. Why?");
    for(unsigned i = 0; i < rows; i++){
        for(unsigned j = 0; j < cols; j++)
            trans[j][i] = array[i][j];
    }
    return 1;
}


/* multiply two matrices together.
 * usage example: multiply(3, 5, 5, 3, A, B, C);
 */
int multiply(unsigned a_rows, unsigned a_cols, unsigned b_rows, unsigned b_cols, double **A, double **B, double **C){
    if(!a_rows || !a_cols || !b_rows || !b_cols)
        return -1;
    assert(A != NULL && B != NULL && C != NULL && "multiply(): pointer is NULL. Why?");
    assert(a_cols == b_rows && "multiply(): matrices are not compatible for multiplication. Why?");
    assert((b_cols < UINT_MAX / a_rows) && "multiply(): unsigned wrap happened. Why?");
    for(unsigned i = 0; i < a_rows; i++){
        for(unsigned j = 0; j < b_cols; j++){
            C[i][j] = 0.0;
            for(unsigned k = 0; k < a_cols; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
    return 1;
}


/* multiply a matrix by a vector.
 * example usage: multiply_x_by_v(3, 5, array, v1, v2);
 */
int multiply_x_by_v(unsigned rows, unsigned cols, double **array, double *vector1, double *vector2){
    if(!rows || !cols)
        return -1;
    assert(array != NULL && vector1 != NULL && vector2 != NULL && "multiply_x_by_v(): pointer is NULL. Why?");

    for(unsigned i = 0; i < rows; i++){
        vector2[i] = 0.0;
        for(unsigned k = 0; k < cols; k++)
            vector2[i] += array[i][k] * vector1[k];
    }
    return 1;
}


/* calculate determinant of a square matrix.
 * usage example: determinant(rows, array);
 */
double determinant(unsigned rows, double **array){
    double **minor;
    double sum = 0.0;
    unsigned i, j, k;

    assert(array != NULL && "determinant(): pointer is NULL. Why?");

    // save some time with simpler calculations.
    if(rows == 0) return 0;
    if(rows == 1) return array[0][0];
    if(rows == 2) return array[0][0] * array[1][1] - array[0][1] * array[1][0];

    // malloc space for minor.
    if((minor = malloc((rows - 1) * sizeof(double*))) == NULL){ puts("malloc failed."); exit(1); }
    for(i = 0; i < rows - 1; i++)
        if((minor[i] = malloc((rows - 1) * sizeof(double))) == NULL){ puts("malloc failed."); exit(1); }

    for(k = 0; k < rows; k++){
        // build minor of array[0][k].
        unsigned minor_row = 0, minor_col = 0;
        for(i = 1; i < rows; i++){
            for(j = 0; j < rows; j++){
                if(j == k)
                    continue;
                minor[minor_row][minor_col] = array[i][j];
                minor_col++;
            }
            minor_col = 0;
            minor_row++;
        }
        if(k % 2)
            sum -= array[0][k] * determinant(rows - 1, minor);
        else
            sum += array[0][k] * determinant(rows - 1, minor);
    }

    // free minor.
    for(i = 0; i < rows - 1; i++)
        free(minor[i]);
    free(minor);

    return sum;
}


/* invert a square matrix.
 * usage example: inverse(rows, array);
 */
int invert(unsigned rows, double **array){
    double **matrix_of_minors;
    double **adjugate;
    double **minor;
    unsigned i, p, k, j;

    if(rows == 0)
        return -1;
    assert(array != NULL && "invert(): pointer is NULL. Why?");

    // malloc space for matrix_of_minors.
    if((matrix_of_minors = malloc(rows * sizeof(double*))) == NULL){ puts("malloc failed."); exit(1); }
    for(i = 0; i < rows; i++)
        if((matrix_of_minors[i] = malloc(rows * sizeof(double))) == NULL){ puts("malloc failed."); exit(1); }

    // malloc space for minor.
    if((minor = malloc((rows - 1) * sizeof(double*))) == NULL){ puts("malloc failed."); exit(1); }
    for(i = 0; i < rows - 1; i++)
        if((minor[i] = malloc((rows - 1) * sizeof(double))) == NULL){ puts("malloc failed."); exit(1); }

    // malloc space for adjugate.
    if((adjugate = malloc(rows * sizeof(double*))) == NULL){ puts("malloc failed."); exit(1); }
    for(i = 0; i < rows; i++)
        if((adjugate[i] = malloc(rows * sizeof(double))) == NULL){ puts("malloc failed."); exit(1); }

    // calculate matrix of minors.
    for(p = 0; p < rows; p++){
        for(k = 0; k < rows; k++){
            unsigned minor_row = 0, minor_col = 0;
            for(i = 0; i < rows; i++){
                for(j = 0; j < rows; j++){
                    if(j == k)
                        continue;
                    if(i == p)
                        continue;
                    minor[minor_row][minor_col] = array[i][j];
                    minor_col++;
                    if(minor_col == rows - 1){
                        minor_col = 0;
                        minor_row++;
                    }
                }
            }
            matrix_of_minors[p][k] = determinant(rows - 1, minor);
        }
    }

    // calculate matrix of cofactors: checkerboard the above.
    for(i = 0; i < rows; i++){
        for(j = 0; j < rows; j++){
            if((i + j) % 2)
                matrix_of_minors[i][j] *= -1.0;
        }
    }

    // transpose matrix of cofactors.
    transpose(rows, rows, matrix_of_minors, adjugate);

    // calculate determinant of array.
    double det = determinant(rows, array);

    // calculate inverse of determinant if determinant not zero.
    if(det)
        det = 1.0 / det;

    // multiply adjugate by det.
    scale_x(rows, rows, det, adjugate);

    // copy adjugate over original array.
    copy(rows, rows, adjugate, array);

    // free mem.
    for(i = 0; i < rows; i++)
        free(matrix_of_minors[i]);
    free(matrix_of_minors);

    for(i = 0; i < rows - 1; i++)
        free(minor[i]);
    free(minor);

    for(i = 0; i < rows; i++)
        free(adjugate[i]);
    free(adjugate);

    return 1;
}


double dot(unsigned rows, double *vector1, double *vector2){
    assert(vector1 != NULL && vector2 != NULL && "dot(): pointer is NULL. Why?");
    double sum = 0.0;

    for(unsigned i = 0; i < rows; i++)
        sum += vector1[i] * vector2[i];

    return sum;
}
