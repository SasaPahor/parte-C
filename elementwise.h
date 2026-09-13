/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */
#ifndef OPS_ELEMENTWISE_H
#define OPS_ELEMENTWISE_H

#include "tensor.h"
#include "error.h"

/*
Operazioni aritmetiche elemento per elemento
Richiedono che i tensori abbiano le stesse dimensioni
*/
ErrorCode tf_add(const Tensor *a, const Tensor *b, Tensor **out);
ErrorCode tf_sub(const Tensor *a, const Tensor *b, Tensor **out);
ErrorCode tf_mul(const Tensor *a, const Tensor *b, Tensor **out);

/*
Operazioni di comparazione elemento per elemento
Restituiscono in output 1.0f se vero o 0.0f se falso
*/
ErrorCode tf_less(const Tensor *a, const Tensor *b, Tensor **out);
ErrorCode tf_greater(const Tensor *a, const Tensor *b, Tensor **out);
ErrorCode tf_equal(const Tensor *a, const Tensor *b, Tensor **out);

/*
Operazioni logiche elemento per elemento
*/
ErrorCode tf_and(const Tensor *a, const Tensor *b, Tensor **out);
ErrorCode tf_or(const Tensor *a, const Tensor *b, Tensor **out);
ErrorCode tf_not(const Tensor *a, Tensor **out);

/*
Operazioni di selezione 

Operano in questo modo:
mask[i] == 1.0f -> out[i] = a[i]
mask[i] == 0.0f -> out[i] = b[i]
*/
ErrorCode tf_select(
    const Tensor *b,
    const Tensor *a,
    const Tensor *mask,
    Tensor **out
);

/*
Operazioni elemento per elemento
*/
ErrorCode tf_relu(const Tensor *a, Tensor **out);
ErrorCode tf_min(const Tensor *a, const Tensor *b, Tensor **out);
ErrorCode tf_max(const Tensor *a, const Tensor *b, Tensor **out);

/*
Operazione di riduzione
Ritorna un tensore 1D con un solo elemento contenente la somma di tutti gli elementi di a
*/
ErrorCode tf_sum(const Tensor *a, Tensor **out);

#endif