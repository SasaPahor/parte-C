/*
  Nome: Sasa
  Cognome: Pahor
  Matricola: SM3201535
*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "error.h"
#include "tensor.h"
#include "stack.h"
#include "parser.h"
#include "elementwise.h"
#include "matrix.h"
#include "convolution.h"
#include "io_pgm.h"
#include "io_tensor.h"



/*
  Controlla che lo stack non sia vuoto.
 */
static void require_stack(Stack *stack, int n)
{
    if (stack == NULL || stack->top + 1 < n)
        error_fatal(ERR_STACK_UNDERFLOW, "numero insufficiente di elementi sullo stack");
}


/*
  Controlla che un Value sia un tensore.
 */
static ErrorCode require_tensor(Value *v, Tensor **out)
{
    if (v == NULL)
        return ERR_STACK_UNDERFLOW;

    if (v->type != VAL_TENSOR)
        return ERR_TYPE_MISMATCH;

    *out = v->as.tensor;
    return ERR_NONE;
}


/*
  Controlla che un Value sia una stringa.
 */
static ErrorCode require_string(Value *v, char **out)
{
    if (v == NULL)
        return ERR_STACK_UNDERFLOW;

    if (v->type != VAL_STRING)
        return ERR_TYPE_MISMATCH;

    *out = v->as.str;
    return ERR_NONE;
}


/*
  Inserisce un tensore nello stack avvolgendolo in un Value.
 */
static void push_tensor(Stack *stack, Tensor *t)
{
    Value *v;

    if (t == NULL)
        error_fatal(ERR_OUT_OF_MEMORY, "impossibile creare il risultato");

    v = value_create_tensor(t);

    if (v == NULL) {
        tensor_release(t);
        error_fatal(ERR_OUT_OF_MEMORY, "impossibile creare il Value del tensore");
    }

    tensor_release(t);

    if (!stack_push(stack, v)) {
        value_release(v);
        error_fatal(ERR_OUT_OF_MEMORY, "impossibile inserire il valore nello stack");
    }
}


/*
  Converte la shape contenuta in un tensore 1D
  in un array di size_t.
 
  La shape deve contenere 1 o 2 valori interi positivi.
*/
static ErrorCode get_shape_from_tensor(const Tensor *t,size_t *shape,size_t *ndim)
{
    size_t i;

    if (t == NULL || t->ndim != 1 || t->total_size < 1 || t->total_size > MAX_DIM) {

        return ERR_DIM_MISMATCH;
    }

    *ndim = t->total_size;

    for (i = 0; i < *ndim; i++) {
        float x = t->data[i];

        if (!isfinite(x) ||
            x <= 0.0f ||
            floorf(x) != x ||
            x > (float)SIZE_MAX) {

            return ERR_DIM_MISMATCH;
        }

        shape[i] = (size_t)x;
    }
    return ERR_NONE;
}


/*
  Implementa l'operazione f:
 
  ( s v -- a )
 
  Crea un tensore della forma s e riempie i suoi elementi
  ripetendo ciclicamente i valori contenuti in v.
 
  Questa operazione realizza il comportamento descritto
  nella specifica del progetto.
*/
static ErrorCode tensor_fill_from_vector(const Tensor *shape_tensor,const Tensor *values,Tensor **out)
{
    size_t shape[MAX_DIM];
    size_t ndim;
    size_t i;
    ErrorCode err;

    if (values == NULL || values->ndim != 1 ||
        values->total_size == 0) {

        return ERR_DIM_MISMATCH;
    }

    err = get_shape_from_tensor(shape_tensor, shape, &ndim);
    if (err != ERR_NONE)
        return err;

    *out = tensor_create(shape, ndim);

    if (*out == NULL)
        return ERR_OUT_OF_MEMORY;

    for (i = 0; i < (*out)->total_size; i++) {
        (*out)->data[i] =
            values->data[i % values->total_size];
    }

    return ERR_NONE;
}


/*
 * Funzione generica per operazioni binarie elemento-per-elemento.
 */
typedef ErrorCode (*BinaryTensorOp)(
    const Tensor *a,
    const Tensor *b,
    Tensor **out
);


static void execute_binary_tensor_op(Stack *stack,BinaryTensorOp operation)
{
    Value *va;
    Value *vb;
    Tensor *a;
    Tensor *b;
    Tensor *result = NULL;
    ErrorCode err;

    require_stack(stack, 2);

    /*
     * La cima è a, sotto troviamo b:
     *
     * ( b a -- result )
     */
    va = stack_pop(stack);
    vb = stack_pop(stack);

    err = require_tensor(va, &a);
    if (err != ERR_NONE) {
        value_release(va);
        value_release(vb);
        error_fatal(err, "errore durante il recupero del tensore a");
    }

    err = require_tensor(vb, &b);
    if (err != ERR_NONE) {
        value_release(va);
        value_release(vb);
        error_fatal(err, "errore durante il recupero del tensore b");
    }

    err = operation(a, b, &result);

    value_release(va);
    value_release(vb);

    if (err != ERR_NONE)
        error_fatal(err, "errore durante l'operazione sui tensori");

    push_tensor(stack, result);
}


/*
 * Operazione di convoluzione:
 *
 * ( a k -- conv(a,k) )
 */
static void execute_conv2d(Stack *stack)
{
    Value *v_kernel;
    Value *v_tensor;
    Tensor *kernel;
    Tensor *tensor;
    Tensor *result = NULL;
    ErrorCode err;

    require_stack(stack, 2);

    v_kernel = stack_pop(stack);
    v_tensor = stack_pop(stack);

    err = require_tensor(v_kernel, &kernel);
    if (err != ERR_NONE) {
        value_release(v_kernel);
        value_release(v_tensor);
        error_fatal(err, "errore durante il recupero del kernel");
    }

    err = require_tensor(v_tensor, &tensor);
    if (err != ERR_NONE) {
        value_release(v_kernel);
        value_release(v_tensor);
        error_fatal(err, "errore durante il recupero del tensore");
    }

    err = tf_conv2d(tensor, kernel, &result);

    value_release(v_kernel);
    value_release(v_tensor);

    if (err != ERR_NONE)
        error_fatal(err, "errore durante la convoluzione");

    push_tensor(stack, result);
}


/*
 * Funzione generica per operazioni unarie.
 */
typedef ErrorCode (*UnaryTensorOp)(
    const Tensor *a,
    Tensor **out
);


static void execute_unary_tensor_op(Stack *stack,UnaryTensorOp operation)
{
    Value *va;
    Tensor *a;
    Tensor *result = NULL;
    ErrorCode err;

    require_stack(stack, 1);

    va = stack_pop(stack);

    err = require_tensor(va, &a);
    if (err != ERR_NONE) {
        value_release(va);
        error_fatal(err, "errore durante il recupero del tensore");
    }

    err = operation(a, &result);

    value_release(va);

    if (err != ERR_NONE)
        error_fatal(err, "errore durante l'operazione sul tensore");

    push_tensor(stack, result);
}


/*
 * Operazione reshape:
 *
 * ( a s -- a' )
 */
static void execute_reshape(Stack *stack)
{
    Value *v_shape;
    Value *v_tensor;

    Tensor *shape_tensor;
    Tensor *tensor;

    size_t shape[MAX_DIM];
    size_t ndim;

    ErrorCode err;

    require_stack(stack, 2);

    /*
     * In cima c'è s, sotto c'è a.
     */
    v_shape = stack_pop(stack);
    v_tensor = stack_pop(stack);

    err = require_tensor(v_shape, &shape_tensor);
    if (err != ERR_NONE) {
        value_release(v_shape);
        value_release(v_tensor);
        error_fatal(err, "errore durante il recupero del tensore shape");
    }

    err = require_tensor(v_tensor, &tensor);
    if (err != ERR_NONE) {
        value_release(v_shape);
        value_release(v_tensor);
        error_fatal(err, "errore durante il recupero del tensore");
    }

    err = get_shape_from_tensor(shape_tensor, shape, &ndim);
    if (err != ERR_NONE) {
        value_release(v_shape);
        value_release(v_tensor);
        error_fatal(err, "errore durante il recupero della shape");
    }

    if (!tensor_reshape(tensor, shape, ndim)) {
        value_release(v_shape);
        value_release(v_tensor);
        error_fatal(ERR_DIM_MISMATCH,"reshape incompatibile con il numero di elementi");
    }

    /*
     * v_shape non serve più.
     */
    value_release(v_shape);

    /*
     * v_tensor viene rimesso nello stack:
     * stack_push prende la proprietà del Value.
     */
    if (!stack_push(stack, v_tensor)) {
        value_release(v_tensor);

        error_fatal(ERR_OUT_OF_MEMORY,"impossibile reinserire il tensore nello stack");
    }
}


/*
 * Operazione ?:
 *
 * ( s -- a )
 */
static void execute_random(Stack *stack)
{
    Value *v_shape;
    Tensor *shape_tensor;

    size_t shape[MAX_DIM];
    size_t ndim;

    Tensor *result;
    ErrorCode err;

    require_stack(stack, 1);

    v_shape = stack_pop(stack);

    err = require_tensor(v_shape, &shape_tensor);
    if (err != ERR_NONE) {
        value_release(v_shape);
        error_fatal(err, "errore durante il recupero del tensore shape");
    }

    err = get_shape_from_tensor(shape_tensor, shape, &ndim);
    if (err != ERR_NONE) {
        value_release(v_shape);
        error_fatal(err, "errore durante il recupero della shape");
    }

    result = tensor_create(shape, ndim);

    value_release(v_shape);

    if (result == NULL)
        error_fatal(ERR_OUT_OF_MEMORY,
                    "impossibile creare il tensore casuale");

    tensor_random(result);

    push_tensor(stack, result);
}


/*
 * Operazione #:
 *
 * ( a -- #a )
 */
static void execute_shape(Stack *stack)
{
    Value *v;
    Tensor *a;
    Tensor *shape;
    ErrorCode err;

    require_stack(stack, 1);

    v = stack_pop(stack);

    err = require_tensor(v, &a);
    if (err != ERR_NONE) {
        value_release(v);
        error_fatal(err, "errore durante il recupero del tensore");
    }

    shape = tensor_get_shape(a);

    value_release(v);

    if (shape == NULL)
        error_fatal(
            ERR_OUT_OF_MEMORY,
            "impossibile creare il tensore shape"
        );

    push_tensor(stack, shape);
}


/*
 * Operazione f:
 *
 * ( s v -- a )
 */
static void execute_fill(Stack *stack)
{
    Value *v_values;
    Value *v_shape;

    Tensor *values;
    Tensor *shape;
    Tensor *result = NULL;

    ErrorCode err;

    require_stack(stack, 2);

    /*
     * In cima: v
     * Sotto: s
     */
    v_values = stack_pop(stack);
    v_shape = stack_pop(stack);

    err = require_tensor(v_values, &values);
    if (err != ERR_NONE) {
        value_release(v_values);
        value_release(v_shape);
        error_fatal(err, "errore durante il recupero del tensore dei valori");
    }

    err = require_tensor(v_shape, &shape);
    if (err != ERR_NONE) {
        value_release(v_values);
        value_release(v_shape);
        error_fatal(err, "errore durante il recupero del tensore shape");
    }

    err = tensor_fill_from_vector(shape, values, &result);

    value_release(v_values);
    value_release(v_shape);

    if (err != ERR_NONE)
        error_fatal(err, "errore durante il fill");

    push_tensor(stack, result);
}


/*
 * Operazione di I/O che scrive un PGM:
 *
 * ( a filename -- )
 */
static void execute_write_pgm(Stack *stack)
{
    Value *v_filename;
    Value *v_tensor;

    Tensor *tensor;
    char *filename;

    ErrorCode err;

    require_stack(stack, 2);

    /*
     * Cima = filename
     * Sotto = a
     */
    v_filename = stack_pop(stack);
    v_tensor = stack_pop(stack);

    err = require_string(v_filename, &filename);
    if (err != ERR_NONE) {
        value_release(v_filename);
        value_release(v_tensor);
        error_fatal(err, "errore durante il recupero del nome del file");
    }

    err = require_tensor(v_tensor, &tensor);
    if (err != ERR_NONE) {
        value_release(v_filename);
        value_release(v_tensor);
        error_fatal(err, "errore durante il recupero del tensore");
    }

    err = tf_write_pgm(tensor, filename);

    value_release(v_filename);
    value_release(v_tensor);

    if (err != ERR_NONE)
        error_fatal(err, "errore nella scrittura del file PGM");
}


/*
 * Operazione di caricamento PGM:
 *
 * ( filename -- tensor )
 */
static void execute_read_pgm(Stack *stack)
{
    Value *v_filename;
    char *filename;

    Tensor *result = NULL;

    ErrorCode err;

    require_stack(stack, 1);

    v_filename = stack_pop(stack);

    err = require_string(v_filename, &filename);
    if (err != ERR_NONE) {
        value_release(v_filename);
        error_fatal(err, "errore durante il recupero del nome del file");
    }

    err = tf_read_pgm(filename, &result);

    value_release(v_filename);

    if (err != ERR_NONE)
        error_fatal(err, "errore nella lettura del file PGM");

    push_tensor(stack, result);
}


/*
 * Operazione di lettura tensor tramite mmap:
 *
 * ( filename -- tensor )
 */
static void execute_read_tensor(Stack *stack)
{
    Value *v_filename;
    char *filename;

    Tensor *result = NULL;

    ErrorCode err;

    require_stack(stack, 1);

    v_filename = stack_pop(stack);

    err = require_string(v_filename, &filename);
    if (err != ERR_NONE) {
        value_release(v_filename);
        error_fatal(err, "errore durante il recupero del nome del file");
    }

    err = tf_read_tensor_mmap(filename, &result);

    value_release(v_filename);

    if (err != ERR_NONE)
        error_fatal(err, "errore nella lettura del file TensorForth");

    push_tensor(stack, result);
}


/*
 * Operazione di scrittura TensorForth:
 *
 * ( a filename -- )
 */
static void execute_write_tensor(Stack *stack)
{
    Value *v_filename;
    Value *v_tensor;

    Tensor *tensor;
    char *filename;

    ErrorCode err;

    require_stack(stack, 2);

    /*
     * Cima = filename
     * Sotto = tensor
     */
    v_filename = stack_pop(stack);
    v_tensor = stack_pop(stack);

    err = require_string(v_filename, &filename);
    if (err != ERR_NONE) {
        value_release(v_filename);
        value_release(v_tensor);
        error_fatal(err, "errore durante il recupero del nome del file");
    }

    err = require_tensor(v_tensor, &tensor);
    if (err != ERR_NONE) {
        value_release(v_filename);
        value_release(v_tensor);
        error_fatal(err, "errore durante il recupero del tensore");
    }

    err = tf_write_tensor_file(tensor, filename);

    value_release(v_filename);
    value_release(v_tensor);

    if (err != ERR_NONE)
        error_fatal(err, "errore nella scrittura del file TensorForth");
}


/*
 * Operazione p:
 *
 * ( a -- )
 */
static void execute_print(Stack *stack)
{
    Value *v;

    require_stack(stack, 1);

    v = stack_pop(stack);

    if (v->type != VAL_TENSOR) {
        value_release(v);
        error_fatal(ERR_TYPE_MISMATCH, "l'operatore p richiede un tensore");
    }

    tensor_print(v->as.tensor);

    value_release(v);
}


/*
 * Operazione di selezione:
 *
 * ( b a m -- m?a:b )
 */
static void execute_select(Stack *stack)
{
    Value *v_mask;
    Value *v_a;
    Value *v_b;

    Tensor *mask;
    Tensor *a;
    Tensor *b;

    Tensor *result = NULL;

    ErrorCode err;

    require_stack(stack, 3);

    /*
     * cima = mask
     * sotto = a
     * sotto ancora = b
     */
    v_mask = stack_pop(stack);
    v_a = stack_pop(stack);
    v_b = stack_pop(stack);

    err = require_tensor(v_mask, &mask);
    if (err != ERR_NONE) {
        value_release(v_mask);
        value_release(v_a);
        value_release(v_b);
        error_fatal(err, "errore durante il recupero del tensore mask");
    }

    err = require_tensor(v_a, &a);
    if (err != ERR_NONE) {
        value_release(v_mask);
        value_release(v_a);
        value_release(v_b);
        error_fatal(err, "errore durante il recupero del tensore a");
    }

    err = require_tensor(v_b, &b);
    if (err != ERR_NONE) {
        value_release(v_mask);
        value_release(v_a);
        value_release(v_b);
        error_fatal(err, "errore durante il recupero del tensore b");
    }

    err = tf_select(b, a, mask, &result);

    value_release(v_mask);
    value_release(v_a);
    value_release(v_b);

    if (err != ERR_NONE)
        error_fatal(err, "errore nell'operazione di selezione");

    push_tensor(stack, result);
}


/*
 * Esegue un operatore del linguaggio TensorForth.
 */
static void execute_operator(Stack *stack, const char *op)
{
    if (strcmp(op, "+") == 0) {
        execute_binary_tensor_op(stack, tf_add);
    }

    else if (strcmp(op, "-") == 0) {
        execute_binary_tensor_op(stack, tf_sub);
    }

    else if (strcmp(op, "*") == 0) {
        execute_binary_tensor_op(stack, tf_mul);
    }

    else if (strcmp(op, "<") == 0) {
        execute_binary_tensor_op(stack, tf_less);
    }

    else if (strcmp(op, ">") == 0) {
        execute_binary_tensor_op(stack, tf_greater);
    }

    else if (strcmp(op, "=") == 0) {
        execute_binary_tensor_op(stack, tf_equal);
    }

    else if (strcmp(op, "&") == 0) {
        execute_binary_tensor_op(stack, tf_and);
    }

    else if (strcmp(op, "|") == 0) {
        execute_binary_tensor_op(stack, tf_or);
    }

    else if (strcmp(op, "!") == 0) {
        execute_unary_tensor_op(stack, tf_not);
    }

    else if (strcmp(op, "$") == 0) {
        execute_select(stack);
    }

    else if (strcmp(op, "@") == 0) {
        execute_binary_tensor_op(stack, tf_matmul);
    }

    else if (strcmp(op, ".") == 0) {
        execute_binary_tensor_op(stack, tf_dot);
    }

    else if (strcmp(op, "c") == 0) {
        execute_conv2d(stack);
    }

    else if (strcmp(op, "R") == 0) {
        execute_unary_tensor_op(stack, tf_relu);
    }

    else if (strcmp(op, "m") == 0) {
        execute_binary_tensor_op(stack, tf_min);
    }

    else if (strcmp(op, "M") == 0) {
        execute_binary_tensor_op(stack, tf_max);
    }

    else if (strcmp(op, "S") == 0) {
        execute_unary_tensor_op(stack, tf_sum);
    }

    /*
     * (a s -- a')
     */
    else if (strcmp(op, "r") == 0) {
        execute_reshape(stack);
    }

    /*
     * (a -- a')
     */
    else if (strcmp(op, "_") == 0) {
        Value *v;

        require_stack(stack, 1);

        v = stack_pop(stack);

        if (v->type != VAL_TENSOR) {
            value_release(v);
            error_fatal(ERR_TYPE_MISMATCH,"ravel richiede un tensore");
        }

        tensor_ravel(v->as.tensor);

        if (!stack_push(stack, v)) {
            value_release(v);
            error_fatal(ERR_OUT_OF_MEMORY,"impossibile reinserire il tensore");
        }
    }

    /*
     * (a -- #a)
     */
    else if (strcmp(op, "#") == 0) {
        execute_shape(stack);
    }

    /*
     * (s -- a)
     */
    else if (strcmp(op, "?") == 0) {
        execute_random(stack);
    }

    /*
     * (s v -- a)
     */
    else if (strcmp(op, "f") == 0) {
        execute_fill(stack);
    }

    /*
     * (a -- )
     */
    else if (strcmp(op, "p") == 0) {
        execute_print(stack);
    }

    /*
     * Operazioni sullo stack.
     */
    else if (strcmp(op, "d") == 0) {
        require_stack(stack, 1);

        if (!stack_dup(stack))
            error_fatal(ERR_OUT_OF_MEMORY,"impossibile duplicare il valore sullo stack");
    }

    else if (strcmp(op, "D") == 0) {
        require_stack(stack, 1);
        stack_drop(stack);
    }

    else if (strcmp(op, "s") == 0) {
        require_stack(stack, 2);
        stack_swap(stack);
    }

    else if (strcmp(op, "o") == 0) {
        require_stack(stack, 2);

        if (!stack_over(stack))
            error_fatal(ERR_OUT_OF_MEMORY, "impossibile eseguire over sullo stack");
    }

    /*
     * I/O
     *
     * ( filename -- tensor )
     */
    else if (strcmp(op, "(") == 0) {
        execute_read_pgm(stack);
    }

    /*
     * ( a filename -- )
     */
    else if (strcmp(op, ")") == 0) {
        execute_write_pgm(stack);
    }

    /*
     * ( filename -- tensor )
     */
    else if (strcmp(op, "{") == 0) {
        execute_read_tensor(stack);
    }

    /*
     * ( a filename -- )
     */
    else if (strcmp(op, "}") == 0) {
        execute_write_tensor(stack);
    }

    else {
        error_fatal(ERR_SYNTAX_ERROR, op);
    }
}


int main(int argc, char *argv[])
{
    FILE *fp;
    Stack *stack;

    /*
     * Il programma richiede esattamente un nome di file:
     *
     * tensorforth [nome file sorgente]
     */
    if (argc != 2) {
        fprintf(stderr,
                "Uso: %s <nome_file_sorgente>\n",
                argv[0]);

        return 1;
    }

    fp = fopen(argv[1], "r");

    if (fp == NULL) {
        error_fatal(ERR_FILE_NOT_FOUND,
                    argv[1]);
    }

    /*
     * Creazione dello stack di esecuzione.
     */
    stack = stack_create();

    if (stack == NULL) {
        fclose(fp);

        error_fatal(ERR_OUT_OF_MEMORY,"impossibile creare lo stack");
    }

    /*
     * Ciclo principale dell'interprete:
     *
     * 1. legge un token
     * 2. verifica il tipo
     * 3. esegue l'azione corrispondente
     * 4. libera il token
     */
    while (1) {
        Token tok;
        ErrorCode err;

        err = parser_next_token(fp, &tok);

        if (err != ERR_NONE) {
            token_free(&tok);
            fclose(fp);
            stack_free(stack);
            error_fatal(err, "errore durante il parsing");
        }

        if (tok.type == TOKEN_EOF) {
            token_free(&tok);
            break;
        }


        if (tok.type == TOKEN_TENSOR) {
            Value *v;

            v = value_create_tensor(tok.as.tensor);

            if (v == NULL) {
                token_free(&tok);
                fclose(fp);
                stack_free(stack);

                error_fatal(ERR_OUT_OF_MEMORY, "impossibile creare Value tensor");
            }

            if (!stack_push(stack, v)) {
                value_release(v);
                token_free(&tok);
                fclose(fp);
                stack_free(stack);

                error_fatal(ERR_OUT_OF_MEMORY, "impossibile inserire il tensore nello stack");
            }
        }

        else if (tok.type == TOKEN_STRING) {
            Value *v;

            v = value_create_string(tok.as.str);

            if (v == NULL) {
                token_free(&tok);
                fclose(fp);
                stack_free(stack);

                error_fatal(ERR_OUT_OF_MEMORY, "impossibile creare Value string");
            }

            if (!stack_push(stack, v)) {
                value_release(v);
                token_free(&tok);
                fclose(fp);
                stack_free(stack);

                error_fatal(ERR_OUT_OF_MEMORY, "impossibile inserire la stringa nello stack");
            }
        }

        else if (tok.type == TOKEN_OPERATOR) {
            execute_operator(stack, tok.as.op_str);
        }

        /*
        * Token sconosciuto.
        */
        else {
            token_free(&tok);
            fclose(fp);
            stack_free(stack);

            error_fatal(ERR_SYNTAX_ERROR, "token non riconosciuto");
        }

        token_free(&tok);
    }

    /*
     * stack_free() deve liberare tutti i Value ancora presenti
     * sullo stack e, tramite il reference counting, i Tensor.
     */
    stack_free(stack);
    fclose(fp);

    return 0;
}
