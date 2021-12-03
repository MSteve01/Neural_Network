#include "Header.h"
#include "Layer.cpp"

extern std::function<Matrix<double>(const Matrix<double>&)> sigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&)> dsigmoid_func;

class LayerId {
public:
	LayerId(Layer::type _Layer_type, const std::size_t& _Layer_size) : Layer_type(_Layer_type), Layer_size(_Layer_size) {

	}

	Layer::type Layer_type;
	std::size_t Layer_size;
	double learning_rate = 0.1;

	std::function<Matrix<double>(const Matrix<double>&)> act_func = sigmoid_func;
	std::function<Matrix<double>(const Matrix<double>&)> dact_func = dsigmoid_func;
};