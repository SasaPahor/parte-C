/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#ifndef PARSER_H
#define PARSER_H

#include <stdio.h>
#include "tensor.h"
#include "error.h"

// Tipo di token riconosciuto
typedef enum {
    TOKEN_TENSOR,    /* Tensore letterale [ 1 2 3 ] */
    TOKEN_STRING,    /* Stringa letterale "file.pgm" */
    TOKEN_OPERATOR,  /* Operatore o comando (es. +, r, ?, d, p) */
    TOKEN_EOF        /* Fine del sorgente */
} TokenType;

// Struttura Token
typedef struct {
    TokenType type;
    union {
        Tensor *tensor;
        char *str;
        char op_str[16];
    } as;
} Token;

/* Inizializza il parser leggendo da file o stringa */
void token_free(Token *tok);
ErrorCode parser_next_token(FILE *fp, Token *tok);

#endif /* PARSER_H */