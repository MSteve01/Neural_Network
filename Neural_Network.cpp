#pragma once

#include "Header.h"
#include "Dense.cpp"
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
		}
		layer.push_back(new Dense(copy.layer.back()->get_size()));
	}

	void rand_weight(std::vector<std::pair<const double&,const double&>> setting ) {
		if (setting.size() != layer.size())
			throw "Invalid random weight value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_weight(setting[i]);
			}
		}
	}
	void rand_bias(std::vector<std::pair<const double&, const double&>> setting) {
		if (setting.size() != layer.size() - 1)
			throw "invalid random bias value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_bias(setting[i]);
			}
		}
	}
	void print_weight() {
		std::cout << "======== weight ========\n";
		for (int i = 0; i < layer.size() - 1; i++) {
			if(layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->print_weight();
		}
	}
	void print_value() {
		std::cout << "======== value =========\n";
		for (int i = 0; i < layer.size(); i++) {
			if(layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->print_value();
		}
	}
	void print_bias() {
		std::cout << "========= bias ==========\n";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->print_bias();
		}
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
		Matrix<double> error = (target - output) * 2;
		for (int i = layer.size() - 2; i >= 0; i--) {
			error = layer[i]->propagation(error);
		}
	}
private:
	std::vector<Layer*> layer;
};