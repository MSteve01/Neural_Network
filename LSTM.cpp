/*#include "Header.h"
#include "Layer.cpp"

extern std::function<Matrix<double>(const Matrix<double>&)> sigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&)> dsigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&)> tanh_func;
extern std::function<Matrix<double>(const Matrix<double>&)> dtanh_func;
double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2);
void set_Matrix(Matrix<double>& M, double value);

class LSTM : public Layer {
public:
	LSTM() {};
	LSTM(const std::size_t& size, const std::size_t& next,
		std::function<Matrix<double>(const Matrix<double>&)> _Iact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dIact_func = dtanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Fact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dFact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Oact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dOact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Kact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dKact_func = dtanh_func) :
		Iact_func(_Iact_func) ,dIact_func(_dIact_func),
		Fact_func(_Fact_func), dFact_func(_dFact_func), 
		Oact_func(_Oact_func), dOact_func(_dOact_func),
		Kact_func(_Kact_func), dKact_func(_dKact_func) {
		fogot_gate.reconstruct(size, 1);
		output_gate.reconstruct(size, 1);
		input_gate.reconstruct(size, 1);

		xO_weight.reconstruct(size,size);
		xF_weight.reconstruct(size, size);
		xI_weight.reconstruct(size, size);
		xK_weight.reconstruct(size, size);
		hO_weight.reconstruct(size, size);
		hF_weight.reconstruct(size, size);
		hI_weight.reconstruct(size, size);
		hK_weight.reconstruct(size, size);

		Obias.reconstruct(size, 1);
		Fbias.reconstruct(size, 1);
		Ibias.reconstruct(size, 1);
		Kbias.reconstruct(size, 1);

		xO_weight_change.reconstruct(size, size);
		xF_weight_change.reconstruct(size, size);
		xI_weight_change.reconstruct(size, size);
		xK_weight_change.reconstruct(size, size);
		hO_weight_change.reconstruct(size, size);
		hF_weight_change.reconstruct(size, size);
		hI_weight_change.reconstruct(size, size);
		hK_weight_change.reconstruct(size, size);

		Obias_change.reconstruct(size, 1);
		Fbias_change.reconstruct(size, 1);
		Ibias_change.reconstruct(size, 1);
		Kbias_change.reconstruct(size, 1);

		c.push_back(Matrix<double>(size, 1));
		h.push_back(Matrix<double>(size, 1));
	}

	Matrix<double> feed() {
		input_gate = Iact_func((xI_weight * value) + (hI_weight * h.back()) + Ibias);
		fogot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);
		output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);
		Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);
		c.push_back(fogot_gate * c.back() + input_gate * K);
		h.push_back(output_gate * c.back());
		return h.back();
	}
	Matrix<double> propagation(const Matrix<double>& gadient) {
		Matrix<double> error
	}
	
protected:
	Matrix<double> fogot_gate;
	Matrix<double> output_gate;
	Matrix<double> input_gate;

	Matrix<double> xO_weight;
	Matrix<double> xF_weight;
	Matrix<double> xI_weight;
	Matrix<double> xK_weight;
	Matrix<double> hO_weight;
	Matrix<double> hF_weight;
	Matrix<double> hI_weight;
	Matrix<double> hK_weight;

	Matrix<double> Obias;
	Matrix<double> Fbias;
	Matrix<double> Ibias;
	Matrix<double> Kbias;

	Matrix<double> xO_weight_change;
	Matrix<double> xF_weight_change;
	Matrix<double> xI_weight_change;
	Matrix<double> xK_weight_change;
	Matrix<double> hO_weight_change;
	Matrix<double> hF_weight_change;
	Matrix<double> hI_weight_change;
	Matrix<double> hK_weight_change;

	Matrix<double> Obias_change;
	Matrix<double> Fbias_change;
	Matrix<double> Ibias_change;
	Matrix<double> Kbias_change;

	std::vector<Matrix<double>> c;
	std::vector<Matrix<double>> h;

	std::function<Matrix<double>(const Matrix<double>&)> Oact_func = sigmoid_func;
	std::function<Matrix<double>(const Matrix<double>&)> Fact_func = sigmoid_func;
	std::function<Matrix<double>(const Matrix<double>&)> Iact_func = tanh_func;
	std::function<Matrix<double>(const Matrix<double>&)> Kact_func = tanh_func;

	std::function<Matrix<double>(const Matrix<double>&)> dOact_func = dsigmoid_func;
	std::function<Matrix<double>(const Matrix<double>&)> dFact_func = dsigmoid_func;
	std::function<Matrix<double>(const Matrix<double>&)> dIact_func = dtanh_func;
	std::function<Matrix<double>(const Matrix<double>&)> dKact_func = dtanh_func;
};*/