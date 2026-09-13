/*
 * Nome: Sasa
 * Cognome: Pahor
 * Matricola: SM3201535
 */

#include <stdio.h>
#include <stdlib.h>
#include "error.h"

/*
    Funzione per la gestione degli errori fatali.
    Stampa un messaggio di errore e termina il programma con un codice di uscita non nullo.
*/
_Noreturn void error_fatal(ErrorCode code, const char *details) {
    fprintf(stderr, "RUN_TIME ERROR [%d]: ", code);
    
    switch (code) {
        case ERR_STACK_UNDERFLOW:
            fprintf(stderr, "Stack Underflow (elementi insufficienti nello stack)");
            break;
        case ERR_TYPE_MISMATCH:
            fprintf(stderr, "Type Mismatch (tipo di dato non valido per l'operatore)");
            break;
        case ERR_DIM_MISMATCH:
            fprintf(stderr, "Dimension Mismatch (dimensioni dei tensori incompatibili)");
            break;
        case ERR_SYNTAX_ERROR:
            fprintf(stderr, "Syntax Error (formato del sorgente non valido)");
            break;
        case ERR_FILE_NOT_FOUND:
            fprintf(stderr, "File Error (impossibile aprire o leggere il file)");
            break;
        case ERR_OUT_OF_MEMORY:
            fprintf(stderr, "Out Of Memory (allocazione dinamica fallita)");
            break;
        default:
            fprintf(stderr, "Errore generico dell'interprete");
            break;
    }

    if (details) {
        fprintf(stderr, " -> %s", details);
    }
    fprintf(stderr, "\n");

    /* Esce sempre con un codice di errore non nullo come richiesto */
    exit(code != 0 ? code : 1);
}