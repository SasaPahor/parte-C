/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#include "ops_elementwise.h"

#include <stddef.h>

/*
Verifica se due tensori hanno la stessa forma
*/
static int same_shape(const Tensor *a, const Tensor *b) {
    if (a == NULL || b == NULL) {
        return 0;
    }
    if (a->ndim != b->ndim) {
        return 0;
    }
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return 0;
        }
    }
    return 1;
}

/*
Crea un nuovo tensore con la stessa forma di un tensore dato
*/
static ErrorCode create_like(const Tensor *a, Tensor **out) {
    if (a == NULL || out == NULL) {
        return ERR_GENERIC;
    }
    
    *out = tensor_create(a->shape, a->ndim);

    if (*out == NULL)
        return ERR_OUT_OF_MEMORY;

    return ERR_NONE;
}

ErrorCode tf_add(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_sub(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_mul(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_less(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] < b->data[i]) ? 1.0f : 0.0f;
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_greater(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] > b->data[i]) ? 1.0f : 0.0f;
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_equal(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] == b->data[i]) ? 1.0f : 0.0f;
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_and(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] =
            (a->data[i] != 0.0f && b->data[i] != 0.0f) ? 1.0f : 0.0f;
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_or(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] =
            (a->data[i] != 0.0f || b->data[i] != 0.0f) ? 1.0f : 0.0f;
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_not(const Tensor *a, Tensor **out)
{
    if (a == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] == 0.0f) ? 1.0f : 0.0f;
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_select(
    const Tensor *b,
    const Tensor *a,
    const Tensor *mask,
    Tensor **out)
{
    if (a == NULL || b == NULL || mask == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b) || !same_shape(a, mask)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (mask->data[i] == 1.0f)
            ? a->data[i]
            : b->data[i];
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_relu(const Tensor *a, Tensor **out)
{
    if (a == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] > 0.0f) ? a->data[i] : 0.0f;
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_min(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] < b->data[i])
            ? a->data[i]
            : b->data[i];
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_max(const Tensor *a, const Tensor *b, Tensor **out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    if (!same_shape(a, b)) {
        return ERR_DIM_MISMATCH;
    }

    Tensor *result = NULL;
    ErrorCode err = create_like(a, &result);
    if (err != ERR_NONE) {
        return err;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < a->total_size; i++) {
        result->data[i] = (a->data[i] > b->data[i])
            ? a->data[i]
            : b->data[i];
    }

    *out = result;
    return ERR_NONE;
}

ErrorCode tf_sum(const Tensor *a, Tensor **out)
{
    if (a == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    size_t shape[1] = {1};
    Tensor *result = NULL;
    result = tensor_create(shape, 1);
    
    if (result == NULL)
        return ERR_OUT_OF_MEMORY;

    float sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < a->total_size; i++) {
        sum += a->data[i];
    }

    result->data[0] = sum;
    *out = result;

    return ERR_NONE;
}