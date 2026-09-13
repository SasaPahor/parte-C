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
 *
 * Restituisce in input: a, b che sono i vettori da moltiplicare
 * Restituisce in output: out che è il puntatore al tensore risultato della moltiplicazione, che contiene la somma di tutti gli elementi di a[i] * b[i]
 */
ErrorCode tf_dot(const Tensor *a, const Tensor *b, Tensor **out);

/*
 * Moltiplicazione tra matrici 2D.
 *
 * Restituisce in input: a, b che sono le matrici da moltiplicare
 * Restituisce in output: out che è il puntatore al tensore risultato della moltiplicazione, che contiene la matrice prodotto di a e b
 * La funzione richiede che il numero di colonne di a sia uguale al numero di righe di b, altrimenti restituisce un errore di dimensione
 */
ErrorCode tf_matmul(const Tensor *a, const Tensor *b, Tensor **out);

#endif /* OPS_MATRIX_H */