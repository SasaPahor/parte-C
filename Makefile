# Nome: Sasa
# Cognome : Pahor
# Matricola: SM3201535

CC = gcc

CFLAGS = -Wall -Wextra -std=c11 -O2 -fopenmp

LDFLAGS = -fopenmp

TARGET = tensorforth

SRCS = main.c \
       parser.c \
       stack.c \
       tensor.c \
       error.c \
       elementwise.c \
       matrix.c \
       convolution.c \
       io_pgm.c \
       io_tensor.c

OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run