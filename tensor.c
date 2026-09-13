/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "tensor.h"


/*
 * Crea un nuovo Tensor con la forma specificata.
 *
 * Alloca:
 * - la struttura Tensor;
 * - l'array shape;
 * - il buffer contenente i dati.
 *
 * Il Tensor creato possiede direttamente il proprio buffer dati.
 */
Tensor *tensor_create(const size_t *shape, size_t ndim)
{
    if (ndim == 0 || ndim > MAX_DIM)
        return NULL;

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));

    if (!t)
        return NULL;

    t->ndim = ndim;

    t->shape = (size_t *)malloc(ndim * sizeof(size_t));

    if (!t->shape) {
        free(t);
        return NULL;
    }

    /*
     * Calcola il numero totale di elementi moltiplicando
     * le dimensioni della shape.
     */
    t->total_size = 1;

    for (size_t i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->total_size *= shape[i];
    }

    /*
     * Alloca il buffer che contiene gli elementi del tensore.
     */
    t->data = (float *)malloc(t->total_size * sizeof(float));

    if (!t->data) {
        free(t->shape);
        free(t);
        return NULL;
    }

    /*
     * Il Tensor appena creato possiede inizialmente
     * un riferimento.
     *
     * mmap_base e mmap_size vengono usati solamente
     * per Tensor caricati tramite mmap.
     */
    t->ref_count = 1;
    t->mmap_base = NULL;
    t->mmap_size = 0;
    t->owns_data = 1;

    return t;
}


void tensor_retain(Tensor *t)
{
    if (t)
        t->ref_count++;
}


void tensor_release(Tensor *t)
{
    if (!t)
        return;

    t->ref_count--;

    if (t->ref_count <= 0) {

        /*
         * Un Tensor normale possiede il proprio buffer
         * e quindi lo libera con free().
         */
        if (t->owns_data && t->data) {
            free(t->data);
        }

        /*
         * Un Tensor caricato tramite mmap non possiede
         * direttamente il buffer dati: deve invece
         * rimuovere il mapping del file.
         */
        else if (!t->owns_data && t->mmap_base) {
            munmap(t->mmap_base, t->mmap_size);
        }

        free(t->shape);
        free(t);
    }
}


void tensor_fill(Tensor *t, float val)
{
    if (!t)
        return;

    for (size_t i = 0; i < t->total_size; i++) {
        t->data[i] = val;
    }
}


void tensor_random(Tensor *t)
{
    if (!t)
        return;

    for (size_t i = 0; i < t->total_size; i++) {
        t->data[i] = (float)rand() / (float)RAND_MAX;
    }
}


/*
 * Stampa il tensore nel formato:
 *
 * Tensor(shape=[...], data=[...])
 */
void tensor_print(const Tensor *t)
{
    if (!t)
        return;

    printf("Tensor(shape=[");

    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu%s", t->shape[i],(i < t->ndim - 1) ? " " : "");
    }

    printf("], data=[");

    for (size_t i = 0; i < t->total_size; i++) {
        printf("%.2f%s", t->data[i],(i < t->total_size - 1) ? " " : "");
    }

    printf("])\n");
}


/*
 * Modifica la forma del tensore senza modificare i dati.
 *
 * La nuova shape è valida solamente se il numero totale
 * di elementi rimane invariato.
 */
int tensor_reshape(Tensor *t,const size_t *new_shape,size_t new_ndim)
{
    if (!t || new_ndim == 0 || new_ndim > MAX_DIM)
        return 0;

    /*
     * Calcola il numero di elementi che avrebbe il tensore
     * con la nuova shape.
     */
    size_t new_total = 1;

    for (size_t i = 0; i < new_ndim; i++) {
        new_total *= new_shape[i];
    }

    /*
     * Una reshape è possibile solo se il numero di elementi
     * rimane uguale.
     */
    if (new_total != t->total_size)
        return 0;

    /*
     * realloc permette di modificare la dimensione
     * dell'array che contiene la shape.
     */
    size_t *tmp_shape =
        (size_t *)realloc(t->shape, new_ndim * sizeof(size_t));

    if (!tmp_shape)
        return 0;

    t->shape = tmp_shape;

    for (size_t i = 0; i < new_ndim; i++) {
        t->shape[i] = new_shape[i];
    }

    t->ndim = new_ndim;

    return 1;
}


/*
 * Trasforma il tensore in un tensore monodimensionale
 * mantenendo invariato il buffer dei dati.
 */
void tensor_ravel(Tensor *t)
{
    if (!t)
        return;

    size_t *new_shape =
        (size_t *)realloc(t->shape, sizeof(size_t));

    if (!new_shape)
        return;

    t->shape = new_shape;
    t->shape[0] = t->total_size;
    t->ndim = 1;
}


/*
 * Operatore #:
 * crea un tensore 1D contenente le dimensioni di t.
 *
 * Ad esempio:
 *
 * Tensor con shape [2 3]
 *        |
 *        v
 * Tensor con shape [2] e dati [2 3]
 */
Tensor *tensor_get_shape(const Tensor *t)
{
    if (!t)
        return NULL;

    /*
     * Il numero di elementi del tensore shape
     * corrisponde al numero di dimensioni di t.
     */
    size_t shape_dim = t->ndim;

    Tensor *s = tensor_create(&shape_dim, 1);

    if (!s)
        return NULL;

    /*
     * Copia le dimensioni di t nei dati del nuovo tensore.
     */
    for (size_t i = 0; i < t->ndim; i++) {
        s->data[i] = (float)t->shape[i];
    }

    return s;
}


/*
 * Crea un Value contenente un intero.
 */
Value *value_create_int(int val)
{
    Value *v = (Value *)malloc(sizeof(Value));

    if (!v)
        return NULL;

    v->type = VAL_INT;
    v->as.i_val = val;
    v->ref_count = 1;

    return v;
}


/*
 * Crea un Value contenente un float.
 */
Value *value_create_float(float val)
{
    Value *v = (Value *)malloc(sizeof(Value));

    if (!v)
        return NULL;

    v->type = VAL_FLOAT;
    v->as.f_val = val;
    v->ref_count = 1;

    return v;
}


/*
 * Crea un Value che contiene un Tensor.
 *
 * Il Value mantiene un riferimento al Tensor tramite
 * il reference counting.
 */
Value *value_create_tensor(Tensor *t)
{
    if (!t)
        return NULL;

    Value *v = (Value *)malloc(sizeof(Value));

    if (!v)
        return NULL;

    v->type = VAL_TENSOR;
    v->as.tensor = t;

    /*
     * Il Value mantiene un riferimento al Tensor.
     */
    tensor_retain(t);

    v->ref_count = 1;

    return v;
}


/*
 * Crea un Value contenente una stringa.
 *
 * La stringa viene duplicata, così il Value possiede
 * una propria copia indipendente della stringa originale.
 */
Value *value_create_string(const char *str)
{
    if (!str)
        return NULL;

    Value *v = (Value *)malloc(sizeof(Value));

    if (!v)
        return NULL;

    v->type = VAL_STRING;
    v->as.str = strdup(str);
    v->ref_count = 1;

    if (!v->as.str) {
        free(v);
        return NULL;
    }

    return v;
}


void value_retain(Value *v)
{
    if (v)
        v->ref_count++;
}


/*
 * Decrementa il reference count del Value.
 *
 * Quando il numero di riferimenti arriva a zero,
 * libera anche la risorsa contenuta nel Value:
 * - Tensor -> tensor_release()
 * - stringa -> free()
 */
void value_release(Value *v)
{
    if (!v)
        return;

    v->ref_count--;

    if (v->ref_count <= 0) {

        if (v->type == VAL_TENSOR) {
            tensor_release(v->as.tensor);
        }
        else if (v->type == VAL_STRING) {
            free(v->as.str);
        }

        free(v);
    }
}


/*
 * Stampa il contenuto di un Value in base al suo tipo.
 */
void value_print(const Value *v)
{
    if (!v)
        return;

    switch (v->type) {

        case VAL_INT:
            printf("%d", v->as.i_val);
            break;

        case VAL_FLOAT:
            printf("%f", v->as.f_val);
            break;

        case VAL_STRING:
            printf("\"%s\"", v->as.str);
            break;

        case VAL_TENSOR:
            tensor_print(v->as.tensor);
            break;
    }
}