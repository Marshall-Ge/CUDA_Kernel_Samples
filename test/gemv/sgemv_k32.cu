#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define CEIL(a,b) ((a)+((b)-1))/(b)