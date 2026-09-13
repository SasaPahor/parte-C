/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#ifndef STACK_H
#define STACK_H

#include "tensor.h"

#define INITIAL_STACK_CAPACITY 16

typedef struct {
    Value **data;      // Array di puntatori a Value
    int top;           // Indice dell'elemento in cima (-1 se vuoto)
    int capacity;      // Capacità massima attuale
} Stack;

/* Ciclo di vita */
Stack *stack_create(void);
void stack_free(Stack *s);

/* Operazioni base */
int stack_push(Stack *s, Value *v);
Value *stack_pop(Stack *s);
Value *stack_peek(const Stack *s);

/* Manipolazioni Forth (d, D, s, o) */
int stack_dup(Stack *s);    /* d */
void stack_drop(Stack *s);  /* D */
void stack_swap(Stack *s);  /* s */
int stack_over(Stack *s);   /* o */
void stack_print(const Stack *s);

#endif // STACK_H
