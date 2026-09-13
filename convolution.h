/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#ifndef OPS_CONVOLUTION_H
#define OPS_CONVOLUTION_H

#include "tensor.h"
#include "error.h"

/*
    Calcola la convoluzione 2D tra due tensori 2D.
*/
ErrorCode tf_conv2d(const Tensor *a, const Tensor *k, Tensor **out);

#endif /* OPS_CONVOLUTION_H */