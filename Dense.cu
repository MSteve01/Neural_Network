#pragma once

#include "Header.cuh"
#include "Layer.cu"
#include "LayerId.cu"

#include "Func.cuh"


class Dense : public Layer {
public:
	Dense() { Layer_type = Layer::DENSE; }

	Dense(const std::size_t& size) {
		Layer_type = Layer::DENSE;

		value.reconstruct(size, 1);

		act_func = sigmoid_func;
		dact_func = dsigmoid_func;
	}

	Dense(const std::size_t& size, const std::size_t& next, 
		nvstd::function<Matrix<double>(const Matrix<double>&)> _act_func = sigmoid_func,
		nvstd::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func = dsigmoid_func) {
		Layer_type = Layer::DENSE;

		value.reconstruct(size, 1);
		weight.reconstruct(next, size);
		weight_change.reconstruct(next, size);
		bias.reconstruct(next, 1);
		bias_change.reconstruct(next, 1);

		act_func = _act_func;
		dact_func = _dact_func;
	}

	Dense(const LayerId& set,const std::size_t& next) {
		Layer_type = Layer::DENSE;

		value.reconstruct(set.Layer_size, 1);
		weight.reconstruct(next, set.Layer_size);
		weight_change.reconstruct(next, set.Layer_size);
		bias.reconstruct(next, 1);
		bias_change.reconstruct(next, 1);

		act_func = sigmoid_func;
		dact_func = dsigmoid_func;

		set_Layer(set.setting);	
	}



	Matrix<double> feed()  {																				// feedforward
		if (!weight.is_constructed())	
			throw "Undefined weight";

		v.push_back(value);

		return act_func((weight * value) + bias);
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gradient)  {					// backpropagation
		if (gradient.size() > v.size())
			throw "invalid gradient size for  backpropagation";

		std::vector<Matrix<double>> value_change;															// containing flowing gradient to be retunred
		std::vector<Matrix<double>> doutput;																// containing derivative of output 

		const std::size_t start_pos = v.size() - gradient.size();											// rearrange the gradient In case givven gradient is shorter than memory

		for (int round = 0; round < gradient.size(); round++) {												// loop though each memory relate to givven gradient
			doutput.push_back(dact_func((weight * v[round + start_pos]) + bias, gradient[round]));			// compute derivative of each time step output
		}

		for (int round = 0; round < gradient.size(); round++) {
			int size = weight.get_size();
			int blockPergrid = upper_value(double(size) / 1024);
			int threadPerblock = std::min(size, 1024);
			device_weightchange_computeDENSE << <blockPergrid, threadPerblock >> > (weight_change.value, doutput[round].value, v[round].value, doutput[round].row, value.row, learning_rate);
			cudaDeviceSynchronize();
		}

		for (int round = 0; round < gradient.size(); round++) {												// loop though each time step
			bias_change = bias_change + (doutput[round] * learning_rate);
		}

		for (int round = 0; round < gradient.size(); round++) {												// loop though each time step
			value_change.push_back(Matrix<double>(value.row, 1));										
			set_Matrix(value_change.back(), 0);																

			int blockPergrid = upper_value(double(value.get_size()) / 1024);
			int threadPerblock = std::min(value.get_size(), 1024);
			device_flow_computeDENSE << <blockPergrid, threadPerblock >> > (value_change.back().value, doutput[round].value, weight.value, weight.row, weight.column);
			cudaDeviceSynchronize();
		}

		return value_change;
	}
	


	void fogot(const std::size_t& number) {																	// delete old memory and shift the new memory
		int h = number;
		if (number > v.size())
			h = v.size();
		for (int i = 0; i < v.size() - h; i++) {
			v[i] = v[i + h];
		}
		for (int i = 0; i < h; i++) {
			v.pop_back();
		}
	}

	void fogot_all() {																						// delete all memory
		fogot(v.size());
	}



	void change_dependencies() {																			// change weight and bias
		weight = weight + weight_change;
		bias = bias + bias_change;
	}

	void set_change_dependencies(const double& value) {														// set changing weight and chaing bias to specifc value
		set_Matrix(weight_change, value);
		set_Matrix(bias_change, value);
	}

	void mul_change_dependencies(const double& value) {														// multiply changing weight and ching bias with specific value
		weight_change = weight_change * value;
		bias_change = bias_change * value;
	}

	void set_learning(const double& value) {
		learning_rate = value;
	}
	


	void reconstruct(const std::size_t& size, const std::size_t& next,
		nvstd::function<Matrix<double>(const Matrix<double>&)> _act_func = sigmoid_func,
		nvstd::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func = dsigmoid_func) {
		fogot_all();

		value.reconstruct(size, 1);
		weight.reconstruct(next, size);
		bias.reconstruct(next, 1);

		act_func = _act_func;
		dact_func = _dact_func;
	}

	void reconstruct(const LayerId& set,const size_t& next) {
		fogot_all();

		value.reconstruct(set.Layer_size, 1);
		weight.reconstruct(next, set.Layer_size);
		bias.reconstruct(next, 1);

		act_func = sigmoid_func;
		dact_func = dsigmoid_func;

		set_Layer(set.setting);
	}



	void rand_weight(const double& min, const double& max) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[weight.get_size()];
		for(int i = 0 ;i<weight.get_size();i++)
			r[i] = mapping(rand() % 10000, 0, 10000, min, max);
		cudaMemcpy(weight.value, r, weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_weight(std::pair<const double&, const double&> setting) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[weight.get_size()];
		for(int i = 0 ;i<weight.get_size();i++)
			r[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		cudaMemcpy(weight.value, r, weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}


	void rand_weight(std::function<double()> func) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[weight.get_size()];
		for(int i = 0;i<weight.get_size();i++)
			r[i] = func();
		cudaMemcpy(weight.value, r, weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_weight(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[weight.get_size()];
		for (int i = 0; i < weight.get_size(); i++)
			r[i] = func(value.row, next);
		cudaMemcpy(weight.value, r, weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_bias(const double& min, const double& max) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";

		double* r = new double[bias.get_size()];
		for (int i = 0; i < bias.get_size();i++)
			r[i] = mapping(rand() % 10000, 0, 10000, min, max);
		cudaMemcpy(bias.value, r, bias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_bias(std::pair<const double&, const double&> setting) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";

		double* r = new double[bias.get_size()];
		for (int i = 0; i < bias.get_size(); i++) 
			r[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		cudaMemcpy(bias.value, r, bias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_bias(std::function<double()> func) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";

		double* r = new double[bias.get_size()];
		for(int i =0 ;i<bias.get_size();i++)
			r[i] = func();
		cudaMemcpy(bias.value, r, bias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_bias(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";

		double* r = new double[bias.get_size()];
		for (int i = 0; i < bias.get_size();i++)
			r[i] = func(value.row, next);
		cudaMemcpy(bias.value, r, bias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}



	void print_weight() {
		std::cout << "---------Dense Layer----------\n";
		weight.print();
	}

	void print_value() {
		std::cout << "---------Dense Layer----------\n";
		value.print();
	}

	void print_bias(){
		std::cout << "---------Dense Layer---------\n";
		bias.print();
	}



	nvstd::function<Matrix<double>(const Matrix<double>&)> get_act_func() {
		return act_func;
	}

	nvstd::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dact_func() {
		return dact_func;
	}
	
protected:
private:
	void set_Layer(const std::string& setting) {															// set the layer using command
		int size = setting.size();
		int i = 0;
		while (i < size) {
			std::string command = get_text(setting, i);
			if (command == "act")
				universal_set_func(act_func, setting, i);
			else if (command == "dact")
				universal_set_func(dact_func, setting, i);
			else if (command == "learning_rate")
				set_learning_rate(setting, i);
			else if (command == "")
				;
			else throw "command not found";
		}
	}

	void set_learning_rate(const std::string& str, int& i) {
		double a = get_number(str, i);
		learning_rate = a;
	}



	Matrix<double> weight_change;																			// containing changing weight computed by backpropagation and will be added to weight															
	Matrix<double> bias_change;																				// containing changing buas computed by backpropagation and will be added to bias \

	Matrix<double> weight;																					// containing weight
	Matrix<double> bias;																					// containing bias

	std::function<Matrix<double>(const Matrix<double>&)> act_func;											// activate function
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dact_func;					// derivatives of activate function
};