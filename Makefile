
PYTHON_DEFAULT_INC = -I/usr/include/python3.5m -I/usr/local/include/python3.6m
PYTHON_NUMPY_INC = -I /usr/lib/python3.5/site-packages/numpy/core/include -I /usr/local/lib/python3.6/site-packages/numpy/core/include
CFLAGS = -march=native -mtune=native -shared -pthread -fopenmp -fPIC -fno-trapv -fwrapv -O2 -Wall -fomit-frame-pointer -fno-strict-aliasing

default: projection.so update.so

update.c: update.pyx 
	cython update.pyx

projection.c: projection.pyx
	cython projection.pyx

update.so: update.c
	g++ $(CFLAGS) $(PYTHON_DEFAULT_INC) $(PYTHON_NUMPY_INC)  -o update.so update.c 
	rm update.c

projection.so: projection.c
	gcc $(CFLAGS) $(PYTHON_DEFAULT_INC) $(PYTHON_NUMPY_INC)  -o projection.so projection.c 
	rm projection.c

clean: 
	rm -f projection.so update.so
	rm -f projection.c update.c
