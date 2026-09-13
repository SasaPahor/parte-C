/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */
#include "ops_convolution.h"

#include <stddef.h>

/*
Verifica se due tensori sono compatibili per la convoluzione 2D.

Restituisce in input: a, k che sarebbero i tensori da confrontare
Restituisce in output: 1 se sono compatibili, 0 altrimenti
 */
static int conv2d_compatible(const Tensor *a, const Tensor *k)
{
    if (a == NULL || k == NULL) {
        return 0;
    }

    if (a->ndim != 2 || k->ndim != 2) {
        return 0;
    }

    return 1;
}

/*
Calcola la convoluzione 2D tra due tensori 2D.
Restituisce in input: a, k che sono i tensori da convolvere
Restituisce in output: out che è il puntatore al tensore risultato della convoluzione, che contiene la convoluzione di a e k
 */
ErrorCode tf_conv2d(const Tensor *a, const Tensor *k, Tensor **out)
{
    if (a == NULL || k == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!conv2d_compatible(a, k)) {
        return ERR_DIM_MISMATCH;
    }

    const size_t height = a->shape[0];
    const size_t width = a->shape[1];

    const size_t kernel_height = k->shape[0];
    const size_t kernel_width = k->shape[1];

    const size_t pad_y = kernel_height / 2;
    const size_t pad_x = kernel_width / 2;

    size_t result_shape[2] = {height, width};

    Tensor *result = NULL;
    result = tensor_create(result_shape, 2);

    if (result == NULL)
        return ERR_OUT_OF_MEMORY;

    #pragma omp parallel for collapse(2)
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {

            float sum = 0.0f;

            for (size_t ky = 0; ky < kernel_height; ky++) {
                for (size_t kx = 0; kx < kernel_width; kx++) {

                    long input_y = (long)y + (long)ky - (long)pad_y;
                    long input_x = (long)x + (long)kx - (long)pad_x;

                    if (
                        input_y >= 0 &&
                        input_y < (long)height &&
                        input_x >= 0 &&
                        input_x < (long)width
                    ) {
                        size_t input_index =
                            (size_t)input_y * width + (size_t)input_x;

                        size_t kernel_index =
                            ky * kernel_width + kx;

                        sum += a->data[input_index] * k->data[kernel_index];
                    }
                }
            }

            result->data[y * width + x] = sum;
        }
    }

    *out = result;

    return ERR_NONE;
}