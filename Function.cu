#pragma once
#include "Header.cuh"
#include "Matrix.cu"

int upper_value(const double& a) {
	if (a != int(a))
		return int(a) + 1;
	return int(a);
}


__global__ void device_sigmoid_func(double* value, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double result = double(1) / (double(1) + exp(-value[pos]));
	if (result != result)
		result = 0.000001;

	value[pos] = result;
}

__global__ void device_dsigmoid_func(double* value, const double* gadient, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double result = exp(-value[pos]) / pow(double(1) + exp(-value[pos]), 2.0);
	result *= gadient[pos];
	if (result != result)
		result = 0.000001;

	value[pos] = result;
}

__global__ void device_tanh_func(double* value, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double result = tanh(value[pos]);
	if (result != result)
		result = 0.000001;
	
	value[pos] = result;
}

__global__ void device_dtanh_func(double* value, const double* gadient, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double result = double(1) - pow(tanh(value[pos]), 2.0);
	result *= gadient[pos];
	if (result != result)
		result = 0.000001;

	value[pos] = result;
}

__global__ void device_softmax_func(double* value,const int sum, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double result = value[pos] / sum;
	if (result != result)
		result = 0.000001;

	value[pos] = result;
}

__global__ void device_exp_func(double* value, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	value[pos] = exp(value[pos]);
}

__global__ void device_getsumBin_func(const double* value, double* out, const int size) {
	__shared__ double cpy[1024];

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int id = threadIdx.x;

	if (pos < size) {
		cpy[id] = value[pos];
	}
	__syncthreads();

	for (int i = 0; i < 10; i++) {
		int des = (1 << i);
		if (!(id & ((des << 1) - 1)) && pos + des < size) {
			cpy[id] += cpy[id + des];
		}
		__syncthreads();
	}

	out[blockIdx.x] = cpy[0];
}

__global__ void device_getsumBru_func(double* sum, const double* value, const int size) {
	double result = 0;
	for (int q = 0; q < size; q++) {
		result += value[q];
	}
	(*sum) = result;
}

__global__ void device_getmaxBin_func( const double* value, double* getmax, const int size) {
	__shared__ double cpy[1024];
	int id = threadIdx.x;
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < size) {
		cpy[id] = value[pos];
	}
	__syncthreads();

	for (int i = 0; i < 10; i++) {
		int filter = (1 << i);
		if (!(id & ((filter << 1) - 1)) && pos + filter < size) {
			cpy[id] = max(cpy[id], cpy[id + filter]);
		}
		__syncthreads();
	}

	getmax[blockIdx.x] = cpy[0];
}

__global__ void device_getmaxBru_func(double* getmax, const double* value, const int size) {
	double _max = value[0];
	for (int i = 1; i < size; i++) {
		_max = max(_max, value[i]);
	}

	(*getmax) = _max;
}

__global__ void device_getminBin_func(const double* value, double* getmin,  const int size) {
	__shared__ double cpy[1024];
	int id = threadIdx.x;
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < size) {
		cpy[id] = value[pos];
	}
	__syncthreads();

	for (int i = 0; i < 10; i++) {
		int filter = (1 << i);
		if (!(id & ((filter << 1) - 1)) && pos + filter < size) {
			cpy[id] = min(cpy[id], cpy[id + filter]);
		}
		__syncthreads();
	}

	getmin[blockIdx.x] = cpy[0];
}

__global__ void device_getminBru_func(double* getmin, const double* value, const int size) {
	double _min = value[0];
	for (int i = 1; i < size; i++) {
		_min = min(_min, value[i]);
	}

	(*getmin) = _min;
}

__global__ void device_dsoftmax_func(double* value, const double* gadient,const int sum, const int size) {
	int pos = blockDim.x * blockIdx.x + threadIdx.x;
	if (pos >= size)
		return;

	double result = value[pos] * (sum + 1) * gadient[pos];
	if (result != result)
		result = 0.000001;

	value[pos] = result;
}

__global__ void device_muleach_func(double* result, const double* value1, const double* value2, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= size)
		return;

	result[i] = value1[i] * value2[i];
}

__global__ void device_plus_func(double* result, const double number, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
		result[i] += number;
}

__global__ void device_ccentloss_func(double* result, const double* target, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < size)
		result[pos] = target[pos] * log(result[pos]);
}

__global__ void device_dccentloss_func(double* result, const double* target, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < size) {
		result[pos] = target[pos] / result[pos];
		if (result[pos] != result[pos])
			result[pos] = 0.0000001;
	}
}

__global__ void device_set_matrix(double* value, const int number, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < size) {
		value[pos] = number;
	}
}

void get_sum(double* sum ,const double* value, const int _size) {
	int size = _size;
	int blockPergrid = upper_value(double(size) / 1024);
	int threadPerblock = std::min(size, 1024);
	int pos = 0;
	double* getsum[2];
	cudaMalloc(&(getsum[0]), size * sizeof(double));
	cudaMalloc(&(getsum[1]), size * sizeof(double));
	cudaMemcpy(getsum[0], value, size * sizeof(double), cudaMemcpyDeviceToDevice);
	while (size > 256) {
		device_getsumBin_func << <blockPergrid, threadPerblock >> > (getsum[pos], getsum[1 - pos], size);
		cudaDeviceSynchronize();
		size = blockPergrid;
		blockPergrid = upper_value(double(size) / 1024);
		threadPerblock = std::min(size, 1024);
		pos = 1 - pos;
	}
	double* get_result;
	cudaMalloc(&get_result, sizeof(double));
	device_getsumBru_func << <1, 1 >> > (get_result, getsum[pos], size);
	cudaDeviceSynchronize();
	cudaMemcpy(sum, get_result, sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(getsum[0]);
	cudaFree(getsum[1]);
	cudaFree(get_result);
}

void get_max(double* result, const double* value, const int _size) {
	int size = _size;
	int blockPergrid = upper_value(double(size) / 1024);
	int threadPerblock = std::min(size, 1024);
	int pos = 0;
	double* getmax[2];
	cudaMalloc(&(getmax[0]), size * sizeof(double));
	cudaMalloc(&(getmax[1]), size * sizeof(double));
	cudaMemcpy(getmax[0], value, size * sizeof(double), cudaMemcpyDeviceToDevice);
	while (size > 256) {
		device_getmaxBin_func << <blockPergrid, threadPerblock >> > (getmax[pos], getmax[1 - pos], size);
		cudaDeviceSynchronize();
		size = blockPergrid;
		blockPergrid = upper_value(double(size) / 1024);
		threadPerblock = std::min(size, 1024);
		pos = 1 - pos;
	}

	double* get_result;
	cudaMalloc(&get_result, sizeof(double));
	device_getsumBru_func << <1, 1 >> > (get_result, getmax[pos], size);
	cudaDeviceSynchronize();
	cudaMemcpy(result, get_result, sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(getmax[0]);
	cudaFree(getmax[1]);
	cudaFree(get_result);
}

void get_min(double* result, const double* value, const int _size) {
	int size = _size;
	int blockPergrid = upper_value(double(size) / 1024);
	int threadPerblock = std::min(size, 1024);
	int pos = 0;
	double* getmin[2];
	cudaMalloc(&(getmin[0]), size * sizeof(double));
	cudaMalloc(&(getmin[1]), size * sizeof(double));
	cudaMemcpy(getmin[0], value, size * sizeof(double), cudaMemcpyDeviceToDevice);
	while (size > 256) {
		device_getminBin_func << <blockPergrid, threadPerblock >> > (getmin[pos], getmin[1 - pos], size);
		cudaDeviceSynchronize();
		size = blockPergrid;
		blockPergrid = upper_value(double(size) / 1024);
		threadPerblock = std::min(size, 1024);
		pos = 1 - pos;
	}
	double* get_result;
	cudaMalloc(&get_result, sizeof(double));
	device_getsumBru_func << <1, 1 >> > (get_result, getmin[pos], size);
	cudaDeviceSynchronize();
	cudaMemcpy(result, get_result, sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(getmin[0]);
	cudaFree(getmin[1]);
	cudaFree(get_result);
}


std::function<Matrix<double>(const Matrix<double>&)> 
sigmoid_func = [] (const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	cudaMemcpy(result.get_value(), input.get_value(), result.get_sizeb(), cudaMemcpyDeviceToDevice);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_sigmoid_func << <blockPergrid, threadPerblock >> > (result.get_value(), result.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dsigmoid_func = [] (const Matrix<double>& input, const Matrix<double> gadient) {
	Matrix<double> result(input.get_row(), input.get_column());
	cudaMemcpy(result.get_value(), input.get_value(), result.get_sizeb(), cudaMemcpyDeviceToDevice);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_dsigmoid_func << <blockPergrid, threadPerblock >> > (result.get_value(), gadient.get_value(), input.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
tanh_func = [] (const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	cudaMemcpy(result.get_value(), input.get_value(), result.get_sizeb(), cudaMemcpyDeviceToDevice);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_tanh_func << <blockPergrid, threadPerblock >> > (result.get_value(), input.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dtanh_func = [] (const Matrix<double>& input, const Matrix<double>& gadient) {
	Matrix<double> result(input.get_row(), input.get_column());
	cudaMemcpy(result.get_value(), input.get_value(), result.get_sizeb(), cudaMemcpyDeviceToDevice);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_dtanh_func << <blockPergrid, threadPerblock >> > (result.get_value(), gadient.get_value(), input.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
linear_func = [] (const Matrix<double>& input) {
	return input;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dlinear_func = [] (const Matrix<double>& input, const Matrix<double>& gadient) {
	return gadient;
};

std::function<Matrix<double>(const Matrix<double>&)> 
soft_max = [] (const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	double* cpy;
	double sum = 0;
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	cudaMalloc(&cpy, result.get_sizeb());
	cudaMemcpy(cpy, input.get_value(), input.get_sizeb(), cudaMemcpyDeviceToDevice);
	device_exp_func << <blockPergrid, threadPerblock >> > (cpy, input.get_size());
	cudaDeviceSynchronize();

	get_sum(&sum, cpy, input.get_size());

	device_softmax_func << <blockPergrid, threadPerblock >> > (cpy, sum, input.get_size());
	cudaDeviceSynchronize();
	cudaMemcpy(result.get_value(), cpy, result.get_sizeb(), cudaMemcpyDeviceToDevice);

	cudaFree(cpy);
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dsoft_max = [] (const Matrix<double>& input, const Matrix<double>& gadient) {
	Matrix<double> result = soft_max(input);
	double sum;
	get_sum(&sum, result.get_value(), result.get_size());

	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_dsoftmax_func << <blockPergrid, threadPerblock >> > (result.get_value(), gadient.get_value(), sum, result.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
descale_func = [] (const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	cudaMemcpy(result.get_value(), input.get_value(), input.get_sizeb(), cudaMemcpyDeviceToDevice);
	double max_value = 0;
	get_max(&max_value, result.get_value(), result.get_size());
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_plus_func << <blockPergrid, threadPerblock >> > (result.get_value(), -max_value, result.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
ddescale_func = [] (const Matrix<double>& input, const Matrix<double>& gadient) {
	return gadient;
};

std::function<double(const Matrix<double>&, const Matrix<double>&)> 
catagorical_CEnt_loss_func = [] (const Matrix<double>& input, const Matrix<double>& target) {
	double* cpy;
	cudaMalloc(&cpy, input.get_sizeb());
	cudaMemcpy(cpy, input.get_value(), input.get_sizeb(), cudaMemcpyDeviceToDevice);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_ccentloss_func << <blockPergrid, threadPerblock >> > (cpy, target.get_value(), input.get_size());
	cudaDeviceSynchronize();
	double result = 0;
	get_sum(&result, cpy, input.get_size());
	return result;
};

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> 
dcatagorical_CEnt_loss_func = [] (const Matrix<double>& input,const Matrix<double>& target) {
	Matrix<double> result(input.get_row(), input.get_column());
	cudaMemcpy(result.get_value(), input.get_value(), result.get_sizeb(), cudaMemcpyDeviceToDevice);
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_dccentloss_func << <blockPergrid, threadPerblock >> > (result.get_value(), target.get_value(), result.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<double()> 
normal_rand_func = [] () {
	return double(std::rand() % 10000) / 10000;
};



__host__ __device__ double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2) {
	return ((value - min1) / (max1 - min1) * (max2 - min2)) + min2;
}

Matrix<double> mul_each(const Matrix<double>& left, const Matrix<double>& right) { // directly multiply a[i][j] and b[i][j]
	if (left.get_row() != right.get_row() || left.get_column() != right.get_column())
		throw "invalid multiply each elemenet";
	Matrix<double> result(left.get_row(),left.get_column());
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_muleach_func << <blockPergrid, threadPerblock >> > (result.get_value(), left.get_value(), right.get_value(), result.get_size());
	cudaDeviceSynchronize();
	return result;
}

void set_Matrix(Matrix<double>& M, double value) { // set every Matrix's member to specific number
	int blockPergrid = upper_value(double(M.get_size()) / 1024);
	int threadPerblock = std::min(M.get_size(), 1024);
	device_set_matrix << <blockPergrid, threadPerblock >> > (M.get_value(), value, M.get_size());
	cudaDeviceSynchronize();
}

double get_max(const Matrix<double>& M) { // get max value of the Matrix
	double max_value = 0;
	get_max(&max_value, M.get_value(), M.get_size());
	return max_value;
}

double get_min(const Matrix<double>& M) { // get min value of the Matrix
	double min_value = 0;
	get_min(&min_value, M.get_value(), M.get_size());
	return min_value;
}



std::string get_text(const std::string& str, int& i) {
	std::string result;
	while (str[i] != '\0' && str[i] != ':' && str[i] != ' ' && str[i] != ',') {
		result.insert(result.end(), str[i]);
		++i;
	}
	++i;
	return result;
}

double get_number(const std::string& str, int& i) { // change number in string to double
	double result = 0;
	int dot_pos = -1;
	while (str[i] != '\0' && str[i] != ':' && str[i] != ' ' && str[i] != ',') {
		if (dot_pos == -1) {
			if (str[i] == '.')
				dot_pos = 1;
			else {
				result = result * 10 + (str[i] - '0');
			}
		}
		else if (dot_pos != -1) {
			result += double(str[i] - '0') * std::pow(10, -dot_pos);
			dot_pos++;
		}
		++i;
	}
	++i;
	return result;
}



void universal_set_func(std::function<Matrix<double>(const Matrix<double>&)>& func, const std::string& setting, int& i) { // return function from string name (func)
	std::string a = get_text(setting, i);
	if (a == "sigmoid")
		func = sigmoid_func;
	else if (a == "tanh")
		func = tanh_func;
	else if (a == "linear")
		func = linear_func;
	else if (a == "soft_max")
		func = soft_max;
	else throw "function not found";
}

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)>& func, const std::string& setting, int& i) { // return function from string name (dfunc)
	std::string a = get_text(setting, i);
	if (a == "dsigmoid")
		func = dsigmoid_func;
	else if (a == "dtanh")
		func = dtanh_func;
	else if (a == "dlinear")
		func = dlinear_func;
	else if (a == "dsoft_max")
		func = dsoft_max;
	else throw "function not found";
}

__global__ void device_weightchange_computeLSTM(double* weight_change, const double* dgate, const double* v, const int weight_row, const int weight_column, const double learning_rate) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int r = pos / weight_column;
	int c = pos % weight_row;
	if (r < weight_row && c < weight_column) {
		weight_change[pos] += dgate[r] * v[c] * learning_rate;
	}
}

__global__ void device_flow_compute(double* flow, const double* dgate, const double* weight, const int weight_row, const int weight_column) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < weight_row) {
		for (int i = 0; i < weight_column; i++) {
			flow[pos] += dgate[pos] * weight[pos * weight_column + i];
		}
	}
}


__global__ void device_weightchange_computeDENSE(double* weight_change, const double* doutput, const double* value, const int doutput_size, const int value_size, const double learning_rate) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int i = pos / value_size;
	int j = pos % value_size;
	if (j < doutput_size && j < value_size) {
		weight_change[pos] += doutput[i] * value[j] * learning_rate;
	}
}

__global__ void device_valuechange_compute(double* value_change, const double* doutput, const double* weight, const int weightrow, const int weightcolumn) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < weightcolumn) {
		for (int i = 0; i < weightrow; i++) {
			value_change[pos] += doutput[i] * weight[i * weightcolumn + pos];
		}
	}
}

__global__ void device_valuechange_compute(double* value_change, const double* gadient, const double* v, const int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		value_change[i] = gadient[i] * v[i];
	}
}