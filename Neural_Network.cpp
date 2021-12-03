#pragma once

#include "Header.h"
#include "Dense.cpp"
#include "LSTM.cpp"
#include "Layerid.cpp"

class Neural_Network {
public:
	Neural_Network() {}
	Neural_Network(std::vector<LayerId> _layer) {
		std::size_t _Layer_size = _layer.size();
		for (int i = 0; i < _Layer_size - 1; i++) {
			if (_layer[i].Layer_type == Layer::type::DENSE) {
				layer.push_back(new Dense(_layer[i].Layer_size, _layer[i + 1].Layer_size,
					_layer[i].act_func, _layer[i].dact_func));
 			}
			else if (_layer[i].Layer_type == Layer::type::LSTM) {
				layer.push_back(new LSTM(_layer[i].Layer_size));
				if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
					throw "invalid Layer afte lstm size";
			}
		}
		if (_layer[_Layer_size - 1].Layer_type != Layer::DENSE)
			throw "the output layer must be Dense layer";
		layer.push_back(new Dense(_layer[_Layer_size - 1].Layer_size));
	}
	Neural_Network(const Neural_Network& copy) {
		for (int i = 0; i < copy.layer.size() - 1; i++) {
			if (copy.layer[i]->get_type() == Layer::DENSE) {
				Dense* _layer = static_cast<Dense*>(copy.layer[i]);
				Dense* _next_layer = static_cast<Dense*>(copy.layer[i + 1]);
				layer.push_back(new Dense(_layer->get_size(), _next_layer->get_size(),
					_layer->get_act_func(), _layer->get_dact_func()));
			}
			else if (copy.layer[i]->get_type() == Layer::LSTM) {
				LSTM* _layer = static_cast<LSTM*>(copy.layer[i]);
				layer.push_back(new LSTM(_layer->get_size()));
			}
		}
		layer.push_back(new Dense(copy.layer.back()->get_size()));
	}
	
	void rand_weight(const std::vector<std::pair<double,double>>& setting ) {
		if (setting.size() != layer.size() - 1)
			throw "Invalid random weight value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_weight(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_weight(setting[i]);
			}
		}
	}
	void rand_bias(const std::vector<std::pair<double,double>>& setting) {
		if (setting.size() != layer.size() - 1)
			throw "invalid random bias value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_bias(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_bias(setting[i]);
			}
		}
	}
	void print_weight() {
		std::cout << "======== weight ========\n";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->print_weight();
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->print_weight();
		}
	}
	void print_value() {
		std::cout << "======== value =========\n";
		for (int i = 0; i < layer.size(); i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->print_value();
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->print_value();
		}
	}
	void print_bias() {
		std::cout << "========= bias ==========\n";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->print_bias();
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->print_bias();
		}
	}
	std::size_t get_layer_size() {
		return layer.size();
	}
	Matrix<double> feedforward(Matrix<double> input) {
		if (input.get_row() != layer[0]->get_size() || input.get_column() != 1)
			throw "invalid input Matrix";
		layer[0]->set_value(input);
		for (int i = 1; i < layer.size(); i++) {
			layer[i]->set_value(layer[i - 1]->feed());
		}
		return static_cast<Dense*>(layer.back())->get_value();
	}
	void backpropagation(Matrix<double> target) {
		Matrix<double> output = static_cast<Dense*>(layer.back())->get_value();
		std::vector<Matrix<double>> error;
		error.push_back((target - output) * 2);
		for (int i = layer.size() - 2; i >= 0; i--) {
			error = layer[i]->propagation(error);
		}
	}
	void fogot(const std::size_t& number) {
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->fogot(number);
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->fogot(number);
		}
	}
	void fogot_all() {
		fogot(layer[0]->v.size());
	}
	void change_dependencies() {
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->change_dependencies();
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->change_dependencies();
		}
	}
	void set_change_dependencies(const double& number) {
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->set_change_dependencies(number);
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->set_change_dependencies(number);
		}
	}
	void set_all_learning_rate(const double& number) {
		for (int i = 0; i < layer.size(); i++) {
			layer[i]->set_learning_rate(number);
		}
	}
	Matrix<double> get_output() {
		return layer.back()->value;
	}
private:
	std::vector<Layer*> layer;
};