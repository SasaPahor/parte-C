/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#ifndef OPS_MATRIX_H
#define OPS_MATRIX_H

#include "tensor.h"
#include "error.h"

/*
 * Prodotto interno tra due vettori 1D.
 */
ErrorCode tf_dot(const Tensor *a, const Tensor *b, Tensor **out);

/*
 * Moltiplicazione riga per colonna tra matrici 2D.
 */
ErrorCode tf_matmul(const Tensor *a, const Tensor *b, Tensor **out);

#endif /* OPS_MATRIX_H */