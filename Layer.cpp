#pragma once

#include "Header.h"
#include "Matrix.cpp"

extern std::function<Matrix<double>(const Matrix<double>&)> sigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dsigmoid_func;
extern class Neural_Network;

class Layer {
public:
	enum type {UNDEFINED,DENSE,RNN,LSTM,DROPOUT,FILTER};
	Layer() {};
	Layer(const type& _Layer_type,const std::size_t& size,const double& _learning_rate = 0.1) {
		Layer_type = _Layer_type;
		value.reconstruct(size, 1);
		learning_rate = _learning_rate;
	}

	std::size_t get_size() {
		return value.get_row();
	}
	Layer::type get_type() {
		return Layer_type;
	}
	double get_learning_rate() {
		return learning_rate;
	}

	virtual Matrix<double> feed() = 0;
	virtual std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient) = 0;
	virtual void fogot(const std::size_t& number) = 0;
	virtual void fogot_all() = 0;
	Matrix<double> get_value() {
		return value;
	}
	void set_learning_rate(const double& _learning_rate) {
		learning_rate = _learning_rate;
	}
	void set_value(const Matrix<double>& _value) {
		value = _value;
	}
	
	Matrix<double> operator=(const Matrix<double>& rhs) {
		return value = rhs;
	}
	
	virtual ~Layer() {}

protected:
	Matrix<double> value; // the last value to pass to another layer
	std::vector<Matrix<double>> v;
	type Layer_type = UNDEFINED;
	double learning_rate = 0.1;
	friend class Neural_Network;
};