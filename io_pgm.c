/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */
#include "io_pgm.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 Converte un valore float in un valore unsigned char per la scrittura in un file PGM binario.

 Restituisce in input: x che è il valore float da convertire
 Restituisce in output: il valore unsigned char corrispondente, con valori limitati a [0, 255]
 */
static unsigned char float_to_pixel(float x)
{
    if (x < 0.0f) {
        x = 0.0f;
    }

    if (x > 1.0f) {
        x = 1.0f;
    }

    return (unsigned char)(x * 255.0f + 0.5f);
}

/*
 Scrive un tensore 2D in un file PGM binario P5.

 Restituisce in input: a che è il tensore da scrivere e filename che è il percorso del file PGM da creare
 Restituisce in output: ERR_NONE se l'operazione è andata a buon fine, altrimenti un codice di errore
 */
ErrorCode tf_write_pgm(const Tensor *a, const char *filename)
{
    if (a == NULL || filename == NULL) {
        return ERR_GENERIC;
    }

    if (a->ndim != 2) {
        return ERR_DIM_MISMATCH;
    }

    const size_t height = a->shape[0];
    const size_t width = a->shape[1];

    FILE *fp = fopen(filename, "wb");

    if (fp == NULL) {
        return ERR_FILE_NOT_FOUND;
    }

    if (fprintf(fp, "P5\n%zu %zu\n255\n", width, height) < 0) {
        fclose(fp);
        return ERR_FILE_NOT_FOUND;
    }

    unsigned char *pixels = (unsigned char *)malloc(a->total_size * sizeof(unsigned char));

    if (pixels == NULL) {
        fclose(fp);
        return ERR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < a->total_size; i++) {
        pixels[i] = float_to_pixel(a->data[i]);
    }

    size_t written = fwrite(
        pixels,
        sizeof(unsigned char),
        a->total_size,
        fp
    );

    free(pixels);

    if (written != a->total_size) {
        fclose(fp);
        return ERR_FILE_NOT_FOUND;
    }

    if (fclose(fp) != 0) {
        return ERR_FILE_NOT_FOUND;
    }

    return ERR_NONE;
}

/*
Legge un'immagine PGM binaria P5 e la converte in un tensore 2D.
Restituisce in input: filename che è il percorso del file PGM da leggere
Restituisce in output: out che è il puntatore al tensore risultato della lettura, che contiene i valori dei pixel normalizzati in [0, 1]
 */
static int read_pgm_token(FILE *fp, char *buffer, size_t buffer_size)
{
    int c;
    size_t len = 0;

    if (fp == NULL || buffer == NULL || buffer_size == 0) {
        return 0;
    }

    /*
     * Salta spazi, newline e commenti.
     * Nei PGM i commenti iniziano con '#'.
     */
    while ((c = fgetc(fp)) != EOF) {
        if (isspace((unsigned char)c)) {
            continue;
        }

        if (c == '#') {
            while ((c = fgetc(fp)) != EOF && c != '\n') {
                /* salta commento */
            }

            continue;
        }

        break;
    }

    if (c == EOF) {
        return 0;
    }

    do {
        if (isspace((unsigned char)c)) {
            break;
        }

        if (len + 1 >= buffer_size) {
            return 0;
        }

        buffer[len] = (char)c;
        len++;

    } while ((c = fgetc(fp)) != EOF);

    buffer[len] = '\0';

    return len > 0;
}


/*
converte una stringa in un valore size_t, restituendo 1 se la conversione è andata a buon fine, 0 altrimenti.

Prende in input: token che è la stringa da convertire
Restituisce in output: out che è il puntatore al valore size_t risultante dalla conversione
 */
static int parse_size_t_token(const char *token, size_t *out)
{
    char *endptr;
    unsigned long value;

    if (token == NULL || out == NULL) {
        return 0;
    }

    errno = 0;
    value = strtoul(token, &endptr, 10);

    if (errno != 0 || endptr == token || *endptr != '\0') {
        return 0;
    }

    *out = (size_t)value;

    return 1;
}

/*
Legge un'immagine PGM binaria P5 e la converte in un tensore 2D.

Prende in input: filename che è il percorso del file PGM da leggere
Restituisce in output: out che è il puntatore al tensore risultato della lettura, che contiene i valori dei pixel normalizzati in [0, 1]
 */
ErrorCode tf_read_pgm(const char *filename, Tensor **out)
{
    if (filename == NULL || out == NULL) {
        return ERR_GENERIC;
    }

    *out = NULL;

    FILE *fp = fopen(filename, "rb");

    if (fp == NULL) {
        return ERR_FILE_NOT_FOUND;
    }

    char token[64];

    /*
     * Magic number: deve essere P5.
     */
    if (!read_pgm_token(fp, token, sizeof(token))) {
        fclose(fp);
        return ERR_SYNTAX_ERROR;
    }

    if (strcmp(token, "P5") != 0) {
        fclose(fp);
        return ERR_SYNTAX_ERROR;
    }

    /*
     * Larghezza.
     */
    if (!read_pgm_token(fp, token, sizeof(token))) {
        fclose(fp);
        return ERR_SYNTAX_ERROR;
    }

    size_t width;

    if (!parse_size_t_token(token, &width) || width == 0) {
        fclose(fp);
        return ERR_SYNTAX_ERROR;
    }

    /*
     * Altezza.
     */
    if (!read_pgm_token(fp, token, sizeof(token))) {
        fclose(fp);
        return ERR_SYNTAX_ERROR;
    }

    size_t height;

    if (!parse_size_t_token(token, &height) || height == 0) {
        fclose(fp);
        return ERR_SYNTAX_ERROR;
    }

    /*
     * Valore massimo.
     * Per il progetto gestiamo PGM a 8 bit, quindi maxval = 255.
     */
    if (!read_pgm_token(fp, token, sizeof(token))) {
        fclose(fp);
        return ERR_SYNTAX_ERROR;
    }

    size_t max_value;

    if (!parse_size_t_token(token, &max_value) || max_value != 255) {
        fclose(fp);
        return ERR_SYNTAX_ERROR;
    }

    if (height > SIZE_MAX / width) {
        fclose(fp);
        return ERR_SYNTAX_ERROR;
    }

    size_t shape[2] = {height, width};

    Tensor *result = tensor_create(shape, 2);

    if (result == NULL) {
        fclose(fp);
        return ERR_OUT_OF_MEMORY;
    }

    const size_t pixel_count = height * width;

    unsigned char *pixels = (unsigned char *)malloc(pixel_count * sizeof(unsigned char));

    if (pixels == NULL) {
        tensor_release(result);
        fclose(fp);
        return ERR_OUT_OF_MEMORY;
    }

    size_t read_count = fread(
        pixels,
        sizeof(unsigned char),
        pixel_count,
        fp
    );

    if (read_count != pixel_count) {
        free(pixels);
        tensor_release(result);
        fclose(fp);
        return ERR_FILE_NOT_FOUND;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < pixel_count; i++) {
        result->data[i] = (float)pixels[i] / 255.0f;
    }

    free(pixels);

    if (fclose(fp) != 0) {
        tensor_release(result);
        return ERR_FILE_NOT_FOUND;
    }

    *out = result;

    return ERR_NONE;
}
