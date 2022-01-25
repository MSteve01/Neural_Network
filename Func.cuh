#pragma once

#include "Header.cuh"
#include "Matrix.cu"

int upper_value(const double& a);

__global__ void device_sigmoid_func(double* value, const int size);

__global__ void device_dsigmoid_func(double* value, const double* gradient, const int size);

__global__ void device_tanh_func(double* value, const int size);

__global__ void device_dtanh_func(double* value, const double* gradient, const int size);

__global__ void device_softmax_func(double* value, const int sum, const int size);

__global__ void device_exp_func(double* value, const int size);

__global__ void device_getsumBin_func(const double* value, double* out, const int size);

__global__ void device_getsumBru_func(double* sum, const double* value, const int size);

__global__ void device_getmaxBin_func(const double* value, double* getmax, const int size);

__global__ void device_getmaxBru_func(double* getmax, const double* value, const int size);

__global__ void device_getminBin_func(const double* value, double* getmin, const int size);

__global__ void device_getminBru_func(double* getmin, const double* value, const int size);

__global__ void device_dsoftmax_func(double* value, const double* gradient, const int sum, const int size);

__global__ void device_muleach_func(double* result, const double* value1, const double* value2, const int size);

__global__ void device_plus_func(double* result, const double number, const int size);

__global__ void device_ccentloss_func(double* result, const double* target, const int size);

__global__ void device_dccentloss_func(double* result, const double* target, const int size);

__global__ void device_set_matrix(double* value, const int number, const int size);

void get_sum(double* sum, const double* value, const int _size);

void get_max(double* result, const double* value, const int _size);

void get_min(double* result, const double* value, const int _size);



extern std::function<Matrix<double>(const Matrix<double>&)>
sigmoid_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dsigmoid_func;

extern std::function<Matrix<double>(const Matrix<double>&)>
tanh_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dtanh_func;

extern std::function<Matrix<double>(const Matrix<double>&)>
linear_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dlinear_func;

extern std::function<Matrix<double>(const Matrix<double>&)>
soft_max;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dsoft_max;

extern std::function<Matrix<double>(const Matrix<double>&)>
descale_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
ddescale_func;

extern std::function<double(const Matrix<double>&, const Matrix<double>&)>
catagorical_CEnt_loss_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dcatagorical_CEnt_loss_func;

extern std::function<double()>
normal_rand_func;



double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2);

Matrix<double> mul_each(const Matrix<double>& left, const Matrix<double>& right);

Matrix<double> devide_each(const Matrix<double>& left, const Matrix<double>& right);

Matrix<double> pow_each(const Matrix<double>& left, const double p);

void set_Matrix(Matrix<double>& M, double value);

void set_up_Matrix(Matrix<double>& M, const Matrix<double>& B);

double get_max(const Matrix<double>& M);

double get_min(const Matrix<double>& M);



std::string get_text(const std::string& str, int& i);

double get_number(const std::string& str, int& i);



void universal_set_func(std::function<Matrix<double>(const Matrix<double>&)>& func, const std::string& setting, int& i);

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>& func, const std::string& setting, int& i);

__global__ void device_weightchange_computeLSTM(double* weight_change, const double* dgate, const double* v, const int weight_row, const int weight_column);

__global__ void device_flow_computeLSTM(double* flow, const double* dgate, const double* weight, const int weight_row, const int weight_column);


__global__ void device_weightchange_computeDENSE(double* weight_change, const double* doutput, const double* value, const int doutput_size, const int value_size);

__global__ void device_flow_computeDENSE(double* value_change, const double* doutput, const double* weight, const int weightrow, const int weightcolumn);

__global__ void device_valuechange_computeDROPOUT(double* value_change, const double* gradient, const double* v, const int size);