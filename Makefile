
PYTHON_DEFAULT_INC = -I/usr/include/python3.6m -I/usr/local/include/python3.6m
PYTHON_NUMPY_INC = -I /usr/lib/python3.6/site-packages/numpy/core/include -I /usr/local/lib/python3.6/site-packages/numpy/core/include

PYTHON_DEFAULT_INC = -I/home/tgokey/.local/miniconda3/envs/oFF/include/python3.7m
PYTHON_NUMPY_INC = -I/home/tgokey/.local/miniconda3/envs/oFF/lib/python3.7/site-packages/numpy/core/include 

CFLAGS = -march=native -mtune=native -shared -pthread -fopenmp -fPIC -O2 -Wall -fomit-frame-pointer -fno-strict-aliasing # -fno-trapv -fwrapv 
CYTHONFLAGS=-v -3

default: projection.so update.so

update.c: update.pyx 
	cython $(CYTHONFLAGS) update.pyx

projection.c: projection.pyx
	cython $(CYTHONFLAGS) projection.pyx

update.so: update.c
	g++ $(CFLAGS) $(PYTHON_DEFAULT_INC) $(PYTHON_NUMPY_INC)  -o update.so update.c 
	rm update.c

projection.so: projection.c
	gcc $(CFLAGS) $(PYTHON_DEFAULT_INC) $(PYTHON_NUMPY_INC)  -o projection.so projection.c 
	rm projection.c

clean: 
	rm -f projection.so update.so
	rm -f projection.c update.c
