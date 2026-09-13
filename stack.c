#include <stdio.h>
#include <stdlib.h>
#include "stack.h"


/*
 * Crea e inizializza uno stack vuoto.
 */
Stack *stack_create(void)
{
    Stack *s = (Stack *)malloc(sizeof(Stack));
    if (!s) return NULL;

    s->capacity = INITIAL_STACK_CAPACITY;
    s->top = -1;

    s->data = (Value **)malloc(s->capacity * sizeof(Value *));
    if (!s->data) {
        free(s);
        return NULL;
    }

    return s;
}


/*
 * Libera lo stack e tutti i Value ancora presenti.
 */
void stack_free(Stack *s)
{
    if (!s) return;

    for (int i = 0; i <= s->top; i++) {
        value_release(s->data[i]);
    }

    free(s->data);
    free(s);
}


/*
 * Inserisce un Value in cima allo stack.
 * Se lo stack è pieno, ne raddoppia la capacità.
 */
int stack_push(Stack *s, Value *v)
{
    if (!s || !v) return 0;

    if (s->top + 1 >= s->capacity) {
        int new_cap = s->capacity * 2;

        Value **tmp =
            (Value **)realloc(s->data, new_cap * sizeof(Value *));

        if (!tmp) return 0;

        s->data = tmp;
        s->capacity = new_cap;
    }

    s->top++;
    s->data[s->top] = v;

    return 1;
}


Value *stack_pop(Stack *s)
{
    if (!s || s->top < 0) return NULL;

    Value *v = s->data[s->top];
    s->top--;

    return v;
}


Value *stack_peek(const Stack *s)
{
    if (!s || s->top < 0) return NULL;

    return s->data[s->top];
}


/*
 * Forth d (dup):
 * ( a -- a a )
 *
 * Viene condiviso lo stesso Value tramite reference counting.
 */
int stack_dup(Stack *s)
{
    Value *top_val = stack_peek(s);

    if (!top_val) return 0;

    value_retain(top_val);

    if (!stack_push(s, top_val)) {
        value_release(top_val);
        return 0;
    }

    return 1;
}


/*
 * Forth D (drop):
 * ( a -- )
 */
void stack_drop(Stack *s)
{
    Value *v = stack_pop(s);

    if (v) {
        value_release(v);
    }
}


/*
 * Forth s (swap):
 * ( b a -- a b )
 */
void stack_swap(Stack *s)
{
    if (!s || s->top < 1) return;

    Value *tmp = s->data[s->top];

    s->data[s->top] = s->data[s->top - 1];
    s->data[s->top - 1] = tmp;
}


/*
 * Forth o (over):
 * ( b a -- b a b )
 *
 * Viene condiviso lo stesso Value tramite reference counting.
 */
int stack_over(Stack *s)
{
    if (!s || s->top < 1) return 0;

    Value *second = s->data[s->top - 1];

    value_retain(second);

    if (!stack_push(s, second)) {
        value_release(second);
        return 0;
    }

    return 1;
}


void stack_print(const Stack *s)
{
    if (!s) return;

    printf("STACK (%d/%d): [ ", s->top + 1, s->capacity);

    for (int i = 0; i <= s->top; i++) {
        value_print(s->data[i]);
        printf(" ");
    }

    printf("] <- TOP\n");
}