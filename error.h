/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#ifndef ERROR_H
#define ERROR_H

typedef enum {
    ERR_NONE = 0,
    ERR_STACK_UNDERFLOW,
    ERR_TYPE_MISMATCH,
    ERR_DIM_MISMATCH,
    ERR_SYNTAX_ERROR,
    ERR_FILE_NOT_FOUND,
    ERR_OUT_OF_MEMORY,
    ERR_GENERIC
} ErrorCode;

/*
 * Stampa un messaggio di errore formattato su stderr e interrompe
 * l'esecuzione del programma con codice di uscita diverso da 0.
 */
_Noreturn void error_fatal(ErrorCode code, const char *details);

#endif /* ERROR_H */