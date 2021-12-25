#pragma once

#include "Header.cuh"
#include "Matrix.cu"
#include "Func.cuh"

class Layer {
public:
	enum type {UNDEFINED,DENSE,RNN,LSTM,DROPOUT,FILTER};													// Layer type

	Layer() {};

	Layer(const type& _Layer_type,const std::size_t& size,const double& _learning_rate = 0.1) {
		Layer_type = _Layer_type;
		value.reconstruct(size, 1);
		learning_rate = _learning_rate;
	}

	virtual ~Layer() {}



	virtual Matrix<double> feed() = 0;																		// required feed funtion to feedforward

	virtual std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gradient) = 0;		// required propagation function to backpropagation

	virtual void fogot(const std::size_t& number) = 0;														// required forgot function

	virtual void fogot_all() = 0;																			// required forgot all function



	std::size_t get_size() {
		return value.row;
	}

	Layer::type get_type() {
		return Layer_type;
	}

	double get_learning_rate() {
		return learning_rate;
	}

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
	
protected:
	Matrix<double> value;																					// pointer containing value to be feeded
	std::vector<Matrix<double>> v;																			// vector containing past value
	type Layer_type = UNDEFINED;																			
	double learning_rate = 0.1;

	friend class Neural_Network;
};