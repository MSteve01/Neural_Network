#include "Header.h"
#include "Layer.cpp"

extern std::function<Matrix<double>(const Matrix<double>&)> sigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&)> dsigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&)> tanh_func;
extern std::function<Matrix<double>(const Matrix<double>&)> dtanh_func;
double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2);
void set_Matrix(Matrix<double>& M, double value);
Matrix<double> mul_each(const Matrix<double>& left, const Matrix<double>& right);

class LSTM : public Layer {
public:
	LSTM() { Layer_type = Layer::LSTM; };
	LSTM(const std::size_t& size,
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
		value.reconstruct(size, 1);
		Layer_type = Layer::LSTM;

		xO_weight.reconstruct(size, size);
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

		init_c.reconstruct(size, 1);
		init_h.reconstruct(size, 1);

		init_c_change.reconstruct(size, 1);
		init_h_change.reconstruct(size, 1);
		set_Matrix(init_c, 0);
		set_Matrix(init_h, 0);

		c.push_back(init_c);
		h.push_back(init_h);
	}

	Matrix<double> feed() {
		Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h.back()) + Ibias);
		Matrix<double> fogot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);
		Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);
		Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);

		c.push_back(mul_each(fogot_gate, c.back()) + mul_each(input_gate, K));
		h.push_back(mul_each(output_gate, c.back()));
		v.push_back(value);

		return h.back();
	}
	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient) {
		if (gadient.size() > c.size())
			throw "invalid gaadient  size for  backpropagatoin lstm";

		Matrix<double> dc(value.get_row(), 1); set_Matrix(dc, 0);
		Matrix<double> dh(value.get_row(), 1); set_Matrix(dh, 0);
		Matrix<double> next_dc(value.get_row(), 1);
		Matrix<double> next_dh(value.get_row(), 1);
		std::vector<Matrix<double>> _gadient;
		std::vector<Matrix<double>> flow_gadient; 
		for (int i = 0; i < v.size(); i++) { flow_gadient.push_back(Matrix<double>(value.get_row(), 1)); set_Matrix(flow_gadient.back(), 0); }

		for (int i = 0; i + gadient.size() < v.size(); i++) {
			_gadient.push_back(Matrix<double>(value.get_row(), 1));
			set_Matrix(_gadient.back(),0);
		}
		for (int i = 0; i < gadient.size(); i++) {
			_gadient.push_back(gadient[i]);
		}

		for (int round = v.size() - 1; round >= 0; round--) {
			set_Matrix(next_dc, 0);
			set_Matrix(next_dh, 0);

			Matrix<double> fogot_gate = Fact_func((xF_weight * v[round]) + (hF_weight * h[round]) + Fbias);
			Matrix<double> output_gate = Oact_func((xO_weight * v[round]) + (hO_weight * h[round]) + Obias);
			Matrix<double> input_gate = Iact_func((xI_weight * v[round]) + (hI_weight * h[round]) + Ibias);
			Matrix<double> K = Kact_func((xK_weight * v[round]) + (hK_weight * h[round]) + Kbias);
			
			Matrix<double> dfogot_gate = dFact_func((xF_weight * v[round]) + (hF_weight * h[round]) + Fbias);
			Matrix<double> doutput_gate = dOact_func((xO_weight * v[round]) + (hO_weight * h[round]) + Obias);
			Matrix<double> dinput_gate = dIact_func((xI_weight * v[round]) + (hI_weight * h[round]) + Ibias);
			Matrix<double> dK = dKact_func((xK_weight * v[round]) + (hK_weight * h[round]) + Kbias);

			for (int i = 0; i < value.get_row(); i++) {
				for (int j = 0; j < value.get_row(); j++) {
					xO_weight_change[i][j] += (_gadient[round][i][0] + dh[i][0]) * c[round + 1][i][0] * doutput_gate[i][0] * v[round][j][0] * learning_rate;
					hO_weight_change[i][j] += (_gadient[round][i][0] + dh[i][0]) * c[round + 1][i][0] * doutput_gate[i][0] * h[round][j][0] * learning_rate;
					xI_weight_change[i][j] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * K[i][0] * dinput_gate[i][0] * v[round][j][0] * learning_rate;
					hI_weight_change[i][j] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * K[i][0] * dinput_gate[i][0] * h[round][j][0] * learning_rate;
					xF_weight_change[i][j] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * c[round][i][0] * dfogot_gate[i][0] * v[round][j][0] * learning_rate;
					hF_weight_change[i][j] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * c[round][i][0] * dfogot_gate[i][0] * h[round][j][0] * learning_rate;
					xK_weight_change[i][j] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * fogot_gate[i][0] * dK[i][0] * v[round][j][0] * learning_rate;
					hK_weight_change[i][j] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * fogot_gate[i][0] * dK[i][0] * h[round][j][0] * learning_rate;

					next_dh[i][0] += (_gadient[round][i][0] + dh[i][0]) * c[round + 1][i][0] * doutput_gate[i][0] * hO_weight[i][j];
					next_dh[i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * K[i][0] * dinput_gate[i][0] * hI_weight[i][j];
					next_dh[i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * c[round][i][0] * dfogot_gate[i][0] * hF_weight[i][j];
					next_dh[i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * fogot_gate[i][0] * dK[i][0] * hK_weight[i][j];

					flow_gadient[round][i][0] += (_gadient[round][i][0] + dh[i][0]) * c[round + 1][i][0] * doutput_gate[i][0] * xO_weight[i][j];
					flow_gadient[round][i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * K[i][0] * dinput_gate[i][0] * xI_weight[i][j];
					flow_gadient[round][i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * c[round][i][0] * dfogot_gate[i][0] * xF_weight[i][j];
					flow_gadient[round][i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * fogot_gate[i][0] * dK[i][0] * xK_weight[i][j];
				
					
				}
				Obias_change[i][0] += (_gadient[round][i][0] + dh[i][0]) * c[round + 1][i][0] * doutput_gate[i][0] * learning_rate;
				Ibias_change[i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * K[i][0] * dinput_gate[i][0] * learning_rate;
				Fbias_change[i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * c[round][i][0] * dfogot_gate[i][0] * learning_rate;
				Kbias_change[i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * fogot_gate[i][0] * dK[i][0] * learning_rate;
			
				next_dc[i][0] += ((_gadient[round][i][0] + dh[i][0]) * output_gate[i][0] + dc[i][0]) * fogot_gate[i][0];
				
			}
			dh = next_dh;
			dc = next_dc;
		}

		for (int i = 0; i < value.get_row(); i++) {
			init_h_change[i][0] += dh[i][0];
			init_c_change[i][0] += dc[i][0];
		}

		return flow_gadient;
	}
	
	void fogot(const std::size_t& number) {
		std::size_t _number = number;
		if (number > v.size())
			_number = v.size();
		for (int i = 0; i < v.size() - _number; i++) {
			c[i] = c[i + _number];
			h[i] = h[i + _number];
			v[i] = v[i + _number];
		}
		for (int i = 0; i < _number; i++) {
			v.pop_back();
			h.pop_back();
			c.pop_back();
		}
		if (c.size() == 0) {
			c.push_back(init_c);
			h.push_back(init_h);
		}
	}
	void fogot_all() {
		fogot(v.size());
	}
	void change_dependencies() {
		xO_weight = xO_weight_change + xO_weight;
		xF_weight = xF_weight_change + xF_weight;
		xI_weight = xI_weight_change + xI_weight;
		xK_weight = xK_weight_change + xK_weight;
		hO_weight = hO_weight_change + hO_weight;
		hF_weight = hF_weight_change + hF_weight;
		hI_weight = hI_weight_change + hI_weight;
		hK_weight = hK_weight_change + hK_weight;

		Obias = Obias_change + Obias;
		Fbias = Fbias_change + Fbias;
		Ibias = Ibias_change + Ibias;
		Kbias = Kbias_change + Kbias;

		init_c = init_c + init_c_change;
		init_h = init_h + init_h_change;
	}
	void set_change_dependencies(const double& number) {
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				xO_weight_change[i][j] = number;
				xF_weight_change[i][j] = number;
				xI_weight_change[i][j] = number;
				xK_weight_change[i][j] = number;
				hO_weight_change[i][j] = number;
				hF_weight_change[i][j] = number;
				hI_weight_change[i][j] = number;
				hK_weight_change[i][j] = number;
			}
			Obias_change[i][0] = number;
			Fbias_change[i][0] = number;
			Ibias_change[i][0] = number;
			Kbias_change[i][0] = number;

			init_c_change[i][0] = number;
			init_h_change[i][0] = number;
		}
	}

	void reconstruct(const std::size_t& size, const std::size_t& next,
		std::function<Matrix<double>(const Matrix<double>&)> _Iact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dIact_func = dtanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Fact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dFact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Oact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dOact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Kact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _dKact_func = dtanh_func) {
		value.reconstruct(size, 1);

		Iact_func = _Iact_func;
		Fact_func = _Fact_func;
		Oact_func = _Oact_func;
		Kact_func = _Kact_func;
		dIact_func = _dIact_func;
		dFact_func = _dFact_func;
		dOact_func = _dOact_func;
		dKact_func = _dKact_func;

		xO_weight.reconstruct(size, size);
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

		init_c.reconstruct(size, 1);
		init_h.reconstruct(size, 1);
		set_Matrix(init_c, 0);
		set_Matrix(init_h, 0);

		fogot_all();
		c.push_back(init_c);
		h.push_back(init_h);
	}

	void rand_weight(const double& min,const double& max) {
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				xO_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
				xF_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
				xI_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
				xK_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
				hO_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
				hF_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
				hI_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
				hK_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
			}
		}
	}

	void rand_weight(std::pair<const double&, const double&> setting) {
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				xO_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
				xF_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
				xI_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
				xK_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
				hO_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
				hF_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
				hI_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
				hK_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			}
		}
	}

	void rand_bias(const double& min, const double& max) {
		for (int i = 0; i < value.get_row(); i++) {
			Obias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
			Fbias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
			Ibias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
			Kbias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
		}
	}

	void rand_bias(std::pair<const double&,const double&> setting) {
		for (int i = 0; i < value.get_row(); i++) {
			Obias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			Fbias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			Ibias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			Kbias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		}
	}
	

	std::function<Matrix<double>(const Matrix<double>&)> get_Oact_func() {
		return Oact_func;
	}
	std::function<Matrix<double>(const Matrix<double>&)> get_Fact_func() {
		return Fact_func;
	}
	std::function<Matrix<double>(const Matrix<double>&)> get_Iact_func() {
		return Iact_func;
	}
	std::function<Matrix<double>(const Matrix<double>&)> get_Kact_func() {
		return Kact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&)> get_dOact_func() {
		return dOact_func;
	}
	std::function<Matrix<double>(const Matrix<double>&)> get_dFact_func() {
		return dFact_func;
	}
	std::function<Matrix<double>(const Matrix<double>&)> get_dIact_func() {
		return dIact_func;
	}
	std::function<Matrix<double>(const Matrix<double>&)> get_dKact_func() {
		return dKact_func;
	}
	void print_weight() {
		std::cout << "--------------LSTM Layer----------\n\n";
		print_xI_weight(); std::cout << std::endl;
		print_xF_weight(); std::cout << std::endl;
		print_xK_weight(); std::cout << std::endl;
		print_xO_weight(); std::cout << std::endl;
		print_hI_weight(); std::cout << std::endl;
		print_hF_weight(); std::cout << std::endl;
		print_hK_weight(); std::cout << std::endl;
		print_hO_weight(); std::cout << std::endl;	
	}
	void print_bias() {
		std::cout << "--------------LSTM Layer----------\n\n";
		print_Ibias(); std::cout << std::endl;
		print_Fbias(); std::cout << std::endl;
		print_Kbias(); std::cout << std::endl;
		print_Obias(); std::cout << std::endl;

		print_init();
	}
	void print_value() {
		std::cout << "--------------LSTM Layer----------\n\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << value[i][0] << "    \t";
		}std::cout << std::endl;
	}
protected:
	void print_xO_weight() {
		std::cout << "  -----x-output weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << xO_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}
	void print_xF_weight() {
		std::cout << "  -----x-fogot weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << xF_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}
	void print_xI_weight() {
		std::cout << "  -----x-input weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << xI_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}
	void print_xK_weight() {
		std::cout << "  -----x-k    weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << xK_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}
	void print_hO_weight() {
		std::cout << "  -----h-output weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << hO_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}
	void print_hF_weight() {
		std::cout << "  -----h-fogot weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << hF_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}
	void print_hI_weight() {
		std::cout << "  -----h-input weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << hI_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}
	void print_hK_weight() {
		std::cout << "  -----h-K     weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << hK_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}

	void print_Obias() {
		std::cout << "   ---output bias------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << Obias[i][0] << "    \t";
		}std::cout << std::endl;
	}
	void print_Fbias() {
		std::cout << "   ---fogot bias------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << Fbias[i][0] << "    \t";
		}std::cout << std::endl;
	}
	void print_Ibias() {
		std::cout << "   ---input bias------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << Ibias[i][0] << "    \t";
		}std::cout << std::endl;
	}
	void print_Kbias() {
		std::cout << "   ---K     bias------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << Kbias[i][0] << "    \t";
		}std::cout << std::endl;
	}
	void print_init() {
		std::cout << " -------- init----------\n";
		for (int i = 0; i < init_c.get_row(); i++) {
			for (int j = 0; j < init_c.get_column(); j++) {
				std::cout << init_c[i][j] << "    \t";
			}std::cout << std::endl;
		}

		for (int i = 0; i < init_h.get_row(); i++) {
			for (int j = 0; j < init_h.get_column(); j++) {
				std::cout << init_h[i][j] << "    \t";
			}std::cout << std::endl;
		}
	}
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
	
	Matrix<double> init_c;
	Matrix<double> init_h;

	Matrix<double> Obias_change;
	Matrix<double> Fbias_change;
	Matrix<double> Ibias_change;
	Matrix<double> Kbias_change;

	Matrix<double> init_c_change;
	Matrix<double> init_h_change;
	
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
};