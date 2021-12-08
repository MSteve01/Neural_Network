#pragma once

#include "Header.h"
#include "Layer.cpp"
#include "LayerId.cpp"

// import functions
extern std::function<Matrix<double>(const Matrix<double>&)> sigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&)> tanh_func;
extern std::function<Matrix<double>(const Matrix<double>&)> linear_func;
extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dsigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dtanh_func;
extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dlinear_func;


// declare functions
double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2);
void set_Matrix(Matrix<double>& M, double value);
void universal_set_func(std::function<Matrix<double>(const Matrix<double>&)>& func, const std::string& setting, int& i);
void universal_set_func(std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>& func, const std::string& setting, int& i);
std::string get_text(const std::string& str, int& i);
double get_number(const std::string& str, int& i);



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
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func = dsigmoid_func) {
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

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient)  {					// backpropagation
		if (gadient.size() > v.size())
			throw "invalid gadient size for  backpropagation";

		std::vector<Matrix<double>> value_change;															// containing flowing gadient to be retunred
		std::vector<Matrix<double>> doutput;																// containing derivative of output 

		const std::size_t start_pos = v.size() - gadient.size();											// rearrange the gadient In case givven gadient is shorter than memory

		for (int round = 0; round < gadient.size(); round++) {												// loop though each memory relate to givven gadient
			doutput.push_back(dact_func((weight * v[round + start_pos]) + bias, gadient[round]));			// compute derivative of each time step output
		}

		for (int round = 0; round < gadient.size(); round++) {												// loop though each time step
			for (int i = 0; i < weight.get_row(); i++) {													// loop though every weight
				for (int j = 0; j < weight.get_column(); j++) {
					weight_change[i][j] += doutput[round][i][0] * v[round + start_pos][j][0] * learning_rate;// compute weight change
				}
			}
		}

		for (int round = 0; round < gadient.size(); round++) {												// loop though each time step
			for (int i = 0; i < bias.get_row(); i++) {														// loop though each bias
				bias_change[i][0] += doutput[round][i][0] * learning_rate;									// compute bias change
			}
		}

		for (int round = 0; round < gadient.size(); round++) {												// loop though each time step
			value_change.push_back(Matrix<double>(value.get_row(), 1));										
			set_Matrix(value_change.back(), 0);																

			for (int i = 0; i < weight.get_row(); i++) {													// loop though each weight 
				for (int j = 0; j < weight.get_column(); j++) {
					value_change.back()[j][0] += doutput[round][i][0] * weight[i][j];						// compute flow gadient
				}
			}
		}

		return value_change;
	}
	


	void forgot(const std::size_t& number) {																	// delete old memory and shift the new memory
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

	void forgot_all() {																						// delete all memory
		forgot(v.size());
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
		std::function<Matrix<double>(const Matrix<double>&)> _act_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func = dsigmoid_func) {
		forgot_all();

		value.reconstruct(size, 1);
		weight.reconstruct(next, size);
		bias.reconstruct(next, 1);

		act_func = _act_func;
		dact_func = _dact_func;
	}

	void reconstruct(const LayerId& set,const size_t& next) {
		forgot_all();

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

	void rand_weight(std::function<double()> func) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		for (int i = 0; i < weight.get_row(); i++) {
			for (int j = 0; j < weight.get_column(); j++) {
				weight[i][j] = func();
			}
		}
	}

	void rand_weight(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		for (int i = 0; i < weight.get_row(); i++) {
			for (int j = 0; j < weight.get_column(); j++) {
				weight[i][j] = func(value.get_row(), next);
			}
		}
	}

	void rand_bias(const double& min, const double& max) {
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

	void rand_bias(std::function<double()> func) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";

		for (int i = 0; i < bias.get_row(); i++) {
			bias[i][0] = func();
		}
	}

	void rand_bias(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";

		for (int i = 0; i < bias.get_row(); i++) {
			bias[i][0] = func(value.get_row(), next);
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

	void print_bias(){
		std::cout << "---------Dense Layer---------\n";
		for (int i = 0; i < bias.get_row(); i++) {
			std::cout << bias[i][0] << "    \t";
		}std::cout << std::endl;
	}



	std::function<Matrix<double>(const Matrix<double>&)> get_act_func() {
		return act_func;
	}

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dact_func() {
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