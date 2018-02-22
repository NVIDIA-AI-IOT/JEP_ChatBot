%module tensorNet

%{
    #define SWIG_FILE_WITH_INIT
    #include "tensorNet.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (int DIM1, int* INPLACE_ARRAY1) {(int dim_in,  int* data_in)}
%apply (int DIM1, int* INPLACE_ARRAY1) {(int dim_out, int* data_out)}

%include "tensorNet.h"
