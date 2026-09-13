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
Restituisce in input: a, k che sono i tensori da convolvere
Restituisce in output: out che è il puntatore al tensore risultato della convoluzione, che contiene la convoluzione di a e k
 */
ErrorCode tf_conv2d(const Tensor *a, const Tensor *k, Tensor **out);

#endif /* OPS_CONVOLUTION_H */