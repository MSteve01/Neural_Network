#pragma once

#include "Header.h"
#include "Layer.cpp"

extern std::function<Matrix<double>(const Matrix<double>&)> sigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&)> dsigmoid_func;
double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2);
void set_Matrix(Matrix<double>& M, double value);

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
		std::function<Matrix<double>(const Matrix<double>&)> _act_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dact_func = dsigmoid_func) {
		Layer_type = Layer::DENSE;
		value.reconstruct(size, 1);
		weight.reconstruct(next, size);
		weight_change.reconstruct(next, size);
		bias.reconstruct(next, 1);
		bias_change.reconstruct(next, 1);
		act_func = _act_func;
		dact_func = _dact_func;
	}

	Matrix<double> feed()  {
		if (!weight.is_constructed())
			throw "Undefined weight";
		return act_func((weight * value) + bias);
	}
	Matrix<double> propagation(const Matrix<double>& gadient)  {
		Matrix<double> doutput = dact_func((weight * value) + bias);

		Matrix<double> value_change(value.get_row(), value.get_column());

		for (int i = 0; i < weight.get_row(); i++) {
			for (int j = 0; j < weight.get_column(); j++) {
				weight_change[i][j] = value[j][0] * gadient[i][0] * doutput[i][0] * learning_rate;
			}
		}

		for (int i = 0; i < bias.get_row(); i++) {
			bias_change[i][0] = gadient[i][0] * doutput[i][0] * learning_rate;
		}

		set_Matrix(value_change, 0);
		for (int i = 0; i < weight.get_row(); i++) {
			for (int j = 0; j < weight.get_column(); j++) {
				value_change[j][0] += weight[i][j] * gadient[i][0] * doutput[i][0];
			}
		}

		return value_change;
	}
	void change_dependencies() {
		weight = weight + weight_change;
		bias = bias + bias_change;
	}
	void set_dependencies(const double& value) {
		set_Matrix(weight_change, value);
		set_Matrix(bias_change, value);
	}
	void reconstruct(const std::size_t& size, const std::size_t& next,
		std::function<Matrix<double>(const Matrix<double>&)> _act_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dact_func = dsigmoid_func) {
		value.reconstruct(size, 1);
		weight.reconstruct(next, size);
		bias.reconstruct(next, 1);
		act_func = _act_func;
		dact_func = _dact_func;
	}
	void rand_weight(const double& min, const double& max) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";
		for (int i = 0; i < weight.get_row(); i++) {
			for (int j = 0; j < weight.get_column(); j++) {
				weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
			}
		}
	}
	void rand_weight(std::pair<const double&, const double&> setting) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";
		for (int i = 0; i < weight.get_row(); i++){
			for (int j = 0; j < weight.get_column(); j++) {
				weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			}
		}
	}
	void rand_bais(const double& min, const double& max) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";
		for (int i = 0; i < bias.get_row(); i++) {
			bias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
		}
	}
	void rand_bias(std::pair<const double&, const double&> setting) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";
		for (int i = 0; i < bias.get_row(); i++) {
			bias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		}
	}
	void print_weight() {
		std::cout << "---------Dense Layer----------\n";
		for (int i = 0; i < weight.get_row(); i++) {
			for (int j = 0; j < weight.get_column(); j++) {
				std::cout << weight[i][j] << "    \t";
			}std::cout << std::endl;
		}
	}
	void print_value() {
		std::cout << "---------Dense Layer----------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << value[i][0] << "    \t";
		}std::cout << std::endl;
	}
	void print_bias() {
		std::cout << "---------Dense Layer---------\n";
		for (int i = 0; i < bias.get_row(); i++) {
			std::cout << bias[i][0] << "    \t";
		}std::cout << std::endl;
	}
	std::function<Matrix<double>(const Matrix<double>&)> get_act_func() {
		return act_func;
	}
	std::function<Matrix<double>(const Matrix<double>&)> get_dact_func() {
		return dact_func;
	}
	Matrix<double> get_value() {
		return value;
	}
	
protected:
private:
	Matrix<double> weight_change;
	Matrix<double> bias_change;

	Matrix<double> weight;
	Matrix<double> bias;
	std::function<Matrix<double>(const Matrix<double>&)> act_func;
	std::function<Matrix<double>(const Matrix<double>&)> dact_func;
};