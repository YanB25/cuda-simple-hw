all: helloworld.out matrix-vector.out
CC=nvcc
FLAGS=-arch=compute_60
helloworld.out: helloworld.cu
	${CC} ${FLAGS} -o $@ ${CCFLAGS} $<
matrix-vector.out: matrix-vector.cu
	${CC} ${FLAGS} -o $@ ${CCFLAGS} $<
