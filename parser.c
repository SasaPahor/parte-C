/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "parser.h"
#include "error.h"

void token_free(Token *tok) {
    if (!tok) return;
    if (tok->type == TOKEN_TENSOR && tok->as.tensor) {
        tensor_release(tok->as.tensor);
        tok->as.tensor = NULL;
    } else if (tok->type == TOKEN_STRING && tok->as.str) {
        free(tok->as.str);
        tok->as.str = NULL;
    }
}

/* Salta spazi e ritorni a capo */
static void skip_whitespace(FILE *fp) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (!isspace(c)) {
            ungetc(c, fp);
            break;
        }
    }
}

/* Parsing tensori 1D: formato [ 1 2 3 ] con spazi obbligatori */
static ErrorCode parse_tensor_literal(FILE *fp, Tensor **out)
{
    int next = fgetc(fp);

    /* Verifichiamo lo spazio obbligatorio dopo '[' */
    if (next != ' ' && next != '\t' && next != '\n' && next != '\r') {
        return ERR_SYNTAX_ERROR;
    }

    size_t capacity = 8;
    size_t count = 0;

    float *values = (float *)malloc(capacity * sizeof(float));

    if (!values) {
        return ERR_OUT_OF_MEMORY;
    }

    while (1) {
        skip_whitespace(fp);

        int c = fgetc(fp);

        if (c == EOF) {
            free(values);
            return ERR_SYNTAX_ERROR;
        }

        if (c == ']') {
            break;
        }

        /* Controllo sintattico: vietate le virgole */
        if (c == ',') {
            free(values);
            return ERR_SYNTAX_ERROR;
        }

        ungetc(c, fp);

        float val;

        if (fscanf(fp, "%f", &val) == 1) {

            if (count >= capacity) {
                capacity *= 2;

                float *tmp = (float *)realloc(
                    values,
                    capacity * sizeof(float)
                );

                if (!tmp) {
                    free(values);
                    return ERR_OUT_OF_MEMORY;
                }

                values = tmp;
            }

            values[count++] = val;

        } else {
            free(values);
            return ERR_SYNTAX_ERROR;
        }
    }

    if (count == 0) {
        free(values);
        return ERR_SYNTAX_ERROR;
    }

    size_t shape[1] = { count };

    Tensor *t = tensor_create(shape, 1);

    if (!t) {
        free(values);
        return ERR_OUT_OF_MEMORY;
    }

    memcpy(t->data, values, count * sizeof(float));

    free(values);

    *out = t;

    return ERR_NONE;
}

/* Parsing stringhe: "file.pgm" */
static ErrorCode parse_string_literal(FILE *fp, char **out)
{
    size_t capacity = 32;
    size_t len = 0;

    char *buf = (char *)malloc(capacity);

    if (!buf) {
        return ERR_OUT_OF_MEMORY;
    }

    int c;

    while ((c = fgetc(fp)) != EOF && c != '"') {

        if (len + 1 >= capacity) {

            capacity *= 2;

            char *tmp = (char *)realloc(buf, capacity);

            if (!tmp) {
                free(buf);
                return ERR_OUT_OF_MEMORY;
            }

            buf = tmp;
        }

        buf[len++] = (char)c;
    }

    if (c == EOF) {
        free(buf);
        return ERR_SYNTAX_ERROR;
    }

    buf[len] = '\0';

    *out = buf;

    return ERR_NONE;
}

/* Riconosce e restituisce il prossimo token */
ErrorCode parser_next_token(FILE *fp, Token *tok)
{
    if (fp == NULL || tok == NULL) {
        return ERR_GENERIC;
    }

    memset(tok, 0, sizeof(Token));

    skip_whitespace(fp);

    int c = fgetc(fp);

    if (c == EOF) {
        tok->type = TOKEN_EOF;
        return ERR_NONE;
    }

    if (c == '[') {

        tok->type = TOKEN_TENSOR;

        ErrorCode err = parse_tensor_literal(fp,&tok->as.tensor);

        if (err != ERR_NONE) {
            tok->type = TOKEN_EOF;
            return err;
        }

        return ERR_NONE;
    }

    if (c == '"') {

        tok->type = TOKEN_STRING;

        ErrorCode err = parse_string_literal(fp,&tok->as.str);

        if (err != ERR_NONE) {
            tok->type = TOKEN_EOF;
            return err;
        }

        return ERR_NONE;
    }

    /* Operatore o identificatore singolo/multiplo */
    tok->type = TOKEN_OPERATOR;

    int idx = 0;

    tok->as.op_str[idx++] = (char)c;

    while ((c = fgetc(fp)) != EOF && !isspace(c) && c != '[' && c != '"') {

        if (idx < 15) {
            tok->as.op_str[idx++] = (char)c;
        }
    }

    if (c != EOF && (isspace(c) || c == '[' || c == '"')) {

        ungetc(c, fp);
    }

    tok->as.op_str[idx] = '\0';

    return ERR_NONE;
}