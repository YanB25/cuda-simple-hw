all: helloworld.out matrix-vector.out
CC=nvcc
FLAGS=
helloworld.out: helloworld.cu
	${CC} -o $@ ${CCFLAGS} $<
matrix-vector.out: matrix-vector.cu
	${CC} -o $@ ${CCFLAGS} $<
