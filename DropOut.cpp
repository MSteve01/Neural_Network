#include "Header.h"
#include "Layer.cpp"

extern std::function<Matrix<double>(const Matrix<double>&)> sigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&)> dsigmoid_func;
extern std::function<Matrix<double>(const Matrix<double>&)> tanh_func;
extern std::function<Matrix<double>(const Matrix<double>&)> dtanh_func;
extern std::function<Matrix<double>(const Matrix<double>&)> linear_func;
extern std::function<Matrix<double>(const Matrix<double>&)> dlinear_func;
double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2);
void set_Matrix(Matrix<double>& M, double value);
Matrix<double> mul_each(const Matrix<double>& left, const Matrix<double>& right);

class DropOut : public Layer {
public:
	DropOut () { ; };
	DropOut(const std::size_t& size, std::function<double()> _rand_func = []() {return double(rand() % 10000) / 10000; }) {
		Layer_type = DROPOUT;
		value.reconstruct(size, 1);
		rand_func = _rand_func;
	}
	Matrix<double> feed() {
		Matrix<double> filter(value.get_row(),1);
		for (int i = 0; i < value.get_row(); i++) {
			filter[i][0] = rand_func() < drop_out_rate ? 0 : 1;
		}
		v.push_back(filter);
		return mul_each(value,filter);
	}
	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient) {
		int start_pos = v.size() - gadient.size();
		std::vector<Matrix<double>> result;
		for (int i = 0; i < gadient.size(); i++) {
			result.push_back(Matrix<double>(gadient[i].get_row(),1));
			for (int j = 0; j < gadient[i].get_row(); j++) {
				for (int k = 0; k < gadient[i].get_column(); k++) {
					result[i][j][k] = gadient[i][j][k] * v[start_pos + i][j][k];
				}
			}
		}
		return result;
	}
	void fogot(const std::size_t& number) {
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
	void fogot_all() {
		fogot(v.size());
	}
	void change_dependencies() {
		
	}
	void set_change_dependencies(const double& number) {

	}
	void mul_change_dependencies(const double& number) {

	}
	void reconstruct(const std::size_t& size, std::function<double()> _rand_func = []() {return double(rand() % 10000) / 10000; }) {
		value.reconstruct(size, 1);
		rand_func = _rand_func;

		fogot_all();
	}
	void rand_weight(const double& min, const double& max) {
		
	}

	void rand_weight(std::pair<const double&, const double&> setting) {
		
	}

	void rand_weight(std::function<double()> func) {
	
	}

	void rand_weight(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {

	}

	void rand_bias(const double& min, const double& max) {
		
	}

	void rand_bias(std::pair<const double&, const double&> setting) {
		
	}

	void rand_bias(std::function<double()> func) {
		
	}

	void rand_bias(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {
		
	}

	void set_drop_out_rate(const double& number) {
		drop_out_rate = number;
	}
	void set_rand_func(std::function<double()> _rand_func) {
		rand_func = _rand_func;
	}
	std::function<double()> get_rand_func() {
		return rand_func;
	}
	double get_drop_out_rate() {
		return drop_out_rate;
	}
	void print_value() {
		std::cout << "---------Dense Layer----------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << value[i][0] << "    \t";
		}std::cout << std::endl;
	}
private:
	std::function<double()> rand_func;
	double drop_out_rate = 0.1;
};