## Nome: Sasa
## Cognome: Pahor
## Matricola: SM3201535

# TensorForth

TensorForth è un interprete per un linguaggio di programmazione **stack-based**, sviluppato in C nell'ambito del progetto di Programmazione Avanzata e Parallela.

Il linguaggio opera direttamente su tensori floating-point a precisione singola (`float`) di una o due dimensioni, quindi vettori e matrici. I programmi sono costituiti da una sequenza di token elaborati da sinistra verso destra secondo una logica postfissa, nello stile del linguaggio Forth.

## Funzionamento

L'interprete mantiene uno **stack di `Value`**. I valori possono essere:

- tensori;
- stringhe;
- interi;
- numeri floating-point.

I tensori utilizzano il **reference counting**, in modo che operazioni come `d` (dup) e `o` (over) possano condividere lo stesso tensore senza copiarne i dati.

Il parser legge un token alla volta dal file sorgente. Il `main` interpreta il token e, quando necessario, esegue l'operazione corrispondente.

Le operazioni principali comprendono:

- operazioni elemento-per-elemento: `+`, `-`, `*`, `<`, `>`, `=`, `&`, `|`, `!`;
- selezione: `$`;
- prodotto matriciale: `@`;
- prodotto scalare: `.`;
- convoluzione 2D: `c`;
- ReLU: `R`;
- minimo e massimo: `m`, `M`;
- somma: `S`;
- reshape: `r`;
- ravel: `_`;
- shape: `#`;
- generazione di tensori casuali: `?`;
- filling: `f`;
- stampa: `p`;
- manipolazione dello stack: `d`, `D`, `s`, `o`;
- lettura e scrittura di immagini PGM: `(` e `)`;
- lettura e scrittura del formato TensorForth: `{` e `}`.

Le operazioni numericamente intensive sono implementate con supporto alla **parallelizzazione tramite OpenMP**, come richiesto dal progetto.

## Gestione degli errori

Gli errori vengono rappresentati tramite `ErrorCode`. Sono gestiti, tra gli altri:

- stack vuoto o con un numero insufficiente di operandi;
- tipi incompatibili;
- dimensioni incompatibili;
- errori di sintassi;
- file non accessibili;
- errori di allocazione della memoria.

In caso di errore il programma segnala il problema e termina con un codice diverso da zero, evitando di proseguire in uno stato non valido.

Le risorse allocate vengono rilasciate correttamente. In particolare, il reference counting permette di gestire la condivisione dei tensori tra più `Value`.

## Input e utilizzo

L'eseguibile viene utilizzato dalla riga di comando nella forma:

```text
tensorforth <nome_file_sorgente>
```

Il file sorgente contiene il programma TensorForth da eseguire.

### Esempio

Un semplice programma che duplica un tensore, lo somma a se stesso e stampa il risultato:

```text
[ 5 5 ] d + p
```

Il risultato è:

```text
Tensor(shape=[2], data=[10.00 10.00])
```

### Esempio con un'immagine

Un esempio di programma per applicare un blur a un'immagine PGM è:

```text
"examples/cray-2.pgm" ( [ 5 5 ] [ 0.04 ] f c "examples/cray-2-blurred.pgm" )
```

Il programma:

1. carica l'immagine PGM;
2. crea un kernel `5 x 5` contenente valori `0.04`;
3. applica la convoluzione;
4. salva il risultato in un nuovo file PGM.

## Formato TensorForth su disco

Il formato utilizzato per la serializzazione dei tensori contiene:

- la shape del tensore;
- il numero di dimensioni;
- l'offset dal quale iniziano i dati;
- i dati del tensore memorizzati come `float`.

L'offset dei dati è fissato a **64 byte**, come previsto dalla specifica del progetto.

L'operatore `{` legge il file utilizzando `mmap`, evitando la copia dei dati del tensore in una nuova area di memoria. Il mapping viene mantenuto fino alla liberazione del tensore.

## Compilazione

Il progetto utilizza lo standard **C11** e richiede il supporto a **OpenMP**.

È possibile compilare il progetto tramite il `Makefile` con:

```bash
make
```

Il compilatore deve quindi essere invocato con il supporto OpenMP (`-fopenmp`).

Per rimuovere i file oggetto e l'eseguibile generati:

```bash
make clean
```

## Esecuzione

Dopo la compilazione:

```bash
./tensorforth <nome_file_sorgente>
```

Per esempio:

```bash
./tensorforth examples/test.tf
```

## Ambiente di test

Il progetto è stato sviluppato e testato in un ambiente **Linux Ubuntu recente**, in conformità con le indicazioni della specifica.

L'implementazione utilizza funzionalità POSIX/Linux per la gestione di `mmap`, coerentemente con l'ambiente di test previsto dal progetto. Non è quindi prevista la compatibilità con Windows.

## Standard e tecnologie

- **Linguaggio:** C
- **Standard:** C11
- **Parallelizzazione:** OpenMP
- **Ambiente di test:** Linux / Ubuntu recente
- **Gestione memoria:** allocazione dinamica e reference counting
- **I/O:** file PGM e formato binario TensorForth
- **Memory mapping:** `mmap`

## Struttura generale

Il progetto è organizzato in moduli, ciascuno dedicato a una parte dell'interprete:

- `main.c` — ciclo principale dell'interprete e gestione degli operatori;
- `parser.c/.h` — analisi dei token del linguaggio;
- `tensor.c/.h` — gestione di tensori e `Value`;
- `stack.c/.h` — implementazione dello stack;
- `elementwise.c/.h` — operazioni elemento-per-elemento;
- `matrix.c/.h` — operazioni matriciali;
- `convolution.c/.h` — convoluzione 2D;
- `io_pgm.c/.h` — lettura e scrittura di immagini PGM;
- `io_tensor.c/.h` — lettura e scrittura del formato TensorForth;
- `error.c/.h` — gestione e segnalazione degli errori.

