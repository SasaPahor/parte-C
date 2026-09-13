/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */
#include "ops_matrix.h"

#include <stddef.h>

/*
 * Verifica se due tensori sono compatibili per il prodotto interno (dot product).
 *
 * Restituisce in input: a, b che sarebbero i tensori da confrontare
 * Restituisce in output: 1 se sono compatibili, 0 altrimenti
 */
static int dot_compatible(const Tensor *a, const Tensor *b)
{
    if (a == NULL || b == NULL) {
        return 0;
    }

    if (a->ndim != 1 || b->ndim != 1) {
        return 0;
    }

    if (a->shape[0] != b->shape[0]) {
        return 0;
    }

    return 1;
}

/*
 * Calcola il prodotto interno tra due vettori 1D.
 *
 * Restituisce in input: a, b che sono i vettori da moltiplicare
 * Restituisce in output: out che è il puntatore al tensore risultato della moltiplicazione, che contiene la somma di tutti gli elementi di a[i] * b[i]
 */
ErrorCode tf_dot(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!dot_compatible(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    size_t shape[1] = {1};
    Tensor *result = NULL;
    result = tensor_create(shape, 1);

    if (result == NULL)
        return ERR_OUT_OF_MEMORY;

    float sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < a->total_size; i++) {
        sum += a->data[i] * b->data[i];
    }

    result->data[0] = sum;
    *out = result;

    return ERR_NONE;
}

/*
 * Verifica se due tensori sono compatibili per la moltiplicazione tra matrici.
 *
 * Restituisce in input: a, b che sarebbero i tensori da confrontare
 * Restituisce in output: 1 se sono compatibili, 0 altrimenti
 */
static int matmul_compatible(const Tensor *a, const Tensor *b)
{
    if (a == NULL || b == NULL) {
        return 0;
    }

    if (a->ndim != 2 || b->ndim != 2) {
        return 0;
    }

    if (a->shape[1] != b->shape[0]) {
        return 0;
    }

    return 1;
}

/*
 * Calcola la moltiplicazione tra due matrici 2D.
 *
 * Restituisce in input: a, b che sono le matrici da moltiplicare
 * Restituisce in output: out che è il puntatore al tensore risultato della moltiplicazione, che contiene la matrice prodotto di a e b
 */
ErrorCode tf_matmul(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!matmul_compatible(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    const size_t m = a->shape[0];
    const size_t n = a->shape[1];
    const size_t p = b->shape[1];

    size_t result_shape[2] = {m, p};
    Tensor *result = NULL;
    result = tensor_create(result_shape, 2);

    if (result == NULL)
        return ERR_OUT_OF_MEMORY;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {

            float sum = 0.0f;

            for (size_t k = 0; k < n; k++) {
                sum += a->data[i * n + k] * b->data[k * p + j];
            }

            result->data[i * p + j] = sum;
        }
    }

    *out = result;

    return ERR_NONE;
}