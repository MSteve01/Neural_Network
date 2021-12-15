#pragma once
#include "Header.cuh"
#include "Layer.cu"
#include "LayerId.cu"

#include "Func.cuh"

class LSTM : public Layer {
public:
	LSTM() { Layer_type = Layer::LSTM; };

	LSTM(const std::size_t& size,
		std::function<Matrix<double>(const Matrix<double>&)> _Iact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dIact_func = dtanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Fact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dFact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Oact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dOact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Kact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dKact_func = dtanh_func) :
		Iact_func(_Iact_func) ,dIact_func(_dIact_func),
		Fact_func(_Fact_func), dFact_func(_dFact_func), 
		Oact_func(_Oact_func), dOact_func(_dOact_func),
		Kact_func(_Kact_func), dKact_func(_dKact_func) {
		
		Layer_type = Layer::LSTM;

		value.reconstruct(size, 1);

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

	LSTM(const LayerId& set) {
		value.reconstruct(set.Layer_size, 1);
		Layer_type = Layer::LSTM;

		xO_weight.reconstruct(set.Layer_size, set.Layer_size);
		xF_weight.reconstruct(set.Layer_size, set.Layer_size);
		xI_weight.reconstruct(set.Layer_size, set.Layer_size);
		xK_weight.reconstruct(set.Layer_size, set.Layer_size);
		hO_weight.reconstruct(set.Layer_size, set.Layer_size);
		hF_weight.reconstruct(set.Layer_size, set.Layer_size);
		hI_weight.reconstruct(set.Layer_size, set.Layer_size);
		hK_weight.reconstruct(set.Layer_size, set.Layer_size);

		Obias.reconstruct(set.Layer_size, 1);
		Fbias.reconstruct(set.Layer_size, 1);
		Ibias.reconstruct(set.Layer_size, 1);
		Kbias.reconstruct(set.Layer_size, 1);

		xO_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xF_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xI_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xK_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hO_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hF_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hI_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hK_weight_change.reconstruct(set.Layer_size, set.Layer_size);

		Obias_change.reconstruct(set.Layer_size, 1);
		Fbias_change.reconstruct(set.Layer_size, 1);
		Ibias_change.reconstruct(set.Layer_size, 1);
		Kbias_change.reconstruct(set.Layer_size, 1);

		init_c.reconstruct(set.Layer_size, 1);
		init_h.reconstruct(set.Layer_size, 1);

		init_c_change.reconstruct(set.Layer_size, 1);
		init_h_change.reconstruct(set.Layer_size, 1);

		set_Matrix(init_c, 0);
		set_Matrix(init_h, 0);

		c.push_back(init_c);
		h.push_back(init_h);

		Iact_func = tanh_func;
		Fact_func = sigmoid_func;
		Oact_func = sigmoid_func;
		Kact_func = tanh_func;

		dIact_func = dtanh_func;
		dFact_func = dsigmoid_func;
		dOact_func = dsigmoid_func;
		dKact_func = dtanh_func;

		set_Layer(set.setting);
	}


		
	Matrix<double> feed() {																						// feedforward
		Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h.back()) + Ibias);			// compute input gate
		Matrix<double> fogot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);			// compute forgot gate
		Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);			// compute output gate
		Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);

		c.push_back(mul_each(fogot_gate, c.back()) + mul_each(input_gate, K));									// compute and remember cell state
		h.push_back(mul_each(output_gate, c.back()));															// compute and remember output fo the cell
		v.push_back(value);																						// remember given input

		return h.back();																						// return output
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient) {						// backpropagation
		if (gadient.size() > c.size())
			throw "invalid gaadient  size for  backpropagatoin lstm";

		Matrix<double> dc(value.get_row(), 1);																	// containing error of cell state for each time step
		Matrix<double> dh(value.get_row(), 1);																	// containing error of output for each time step

		Matrix<double> next_dc(value.get_row(), 1);																// containing error of next time step
		Matrix<double> next_dh(value.get_row(), 1);

		set_Matrix(dc, 0);
		set_Matrix(dh, 0);

		std::vector<Matrix<double>> _gadient;
		std::vector<Matrix<double>> flow_gadient;



		for (int i = 0; i < v.size(); i++) {																	// rearrange the gadient and put into _gadient
			flow_gadient.push_back(Matrix<double>(value.get_row(), 1)); set_Matrix(flow_gadient.back(), 0); 
		}

		for (int i = 0; i + gadient.size() < v.size(); i++) {
			_gadient.push_back(Matrix<double>(value.get_row(), 1));
			set_Matrix(_gadient.back(),0);
		}

		for (int i = 0; i < gadient.size(); i++) {
			_gadient.push_back(gadient[i]);
		}



		for (int round = v.size() - 1; round >= 0; round--) {													// loop thougj eery time step
			set_Matrix(next_dc, 0);
			set_Matrix(next_dh, 0);

			Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h[round]) + Ibias);		// compute input gate
			Matrix<double> fogot_gate = Fact_func((xF_weight * value) + (hF_weight * h[round]) + Fbias);		// compute forgot gate
			Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h[round]) + Obias);		// comput output gate
			Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h[round]) + Kbias);

			dh = dh + _gadient[round];																			// add up output error
			dc = dc + mul_each(dh, output_gate);																// add up cell state error

			Matrix<double> dinput_gate = dIact_func((xI_weight * value) + (hI_weight * h[round]) + Ibias, mul_each(dc, K));// derivative of input gate
			Matrix<double> dfogot_gate = dFact_func((xF_weight * value) + (hF_weight * h[round]) + Fbias, mul_each(dc, c[round]));// derivative of forgot gate
			Matrix<double> doutput_gate = dOact_func((xO_weight * value) + (hO_weight * h[round]) + Obias, mul_each(dh, c[round + 1]));// derivative of output
			Matrix<double> dK = dKact_func((xK_weight * value) + (hK_weight * h[round]) + Kbias, mul_each(dc, input_gate));


			int blockPergrid = upper_value(double(value.get_size()) / 1024);
			int threadPerblock = std::min(value.get_size(), 1024);

			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (xO_weight_change.get_value(), doutput_gate.get_value(), v[round].get_value(), xO_weight_change.get_row(), xO_weight_change.get_column(), learning_rate);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (hO_weight_change.get_value(), doutput_gate.get_value(), h[round].get_value(), hO_weight_change.get_row(), hO_weight_change.get_column(), learning_rate);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (xI_weight_change.get_value(), dinput_gate.get_value(), v[round].get_value(), xI_weight_change.get_row(), xI_weight_change.get_column(), learning_rate);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (hI_weight_change.get_value(), dinput_gate.get_value(), h[round].get_value(), hI_weight_change.get_row(), hI_weight_change.get_column(), learning_rate);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (xF_weight_change.get_value(), dfogot_gate.get_value(), v[round].get_value(), xF_weight_change.get_row(), xF_weight_change.get_column(), learning_rate);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (hF_weight_change.get_value(), dfogot_gate.get_value(), h[round].get_value(), hF_weight_change.get_row(), hF_weight_change.get_column(), learning_rate);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (xK_weight_change.get_value(), dK.get_value(), v[round].get_value(), xK_weight_change.get_row(), xK_weight_change.get_column(), learning_rate);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (hK_weight_change.get_value(), dK.get_value(), h[round].get_value(), hK_weight_change.get_row(), hK_weight_change.get_column(), learning_rate);
			cudaDeviceSynchronize();

			device_flow_compute << <blockPergrid, threadPerblock >> > (next_dh.get_value(), doutput_gate.get_value(), hO_weight_change.get_value(), hO_weight_change.get_row(), hO_weight_change.get_column());
			cudaDeviceSynchronize();
			device_flow_compute << <blockPergrid, threadPerblock >> > (next_dh.get_value(), dinput_gate.get_value(), hI_weight_change.get_value(), hI_weight_change.get_row(), hI_weight_change.get_column());
			cudaDeviceSynchronize();
			device_flow_compute << <blockPergrid, threadPerblock >> > (next_dh.get_value(), dfogot_gate.get_value(), hF_weight_change.get_value(), hF_weight_change.get_row(), hF_weight_change.get_column());
			cudaDeviceSynchronize();
			device_flow_compute << <blockPergrid, threadPerblock >> > (next_dh.get_value(), dK.get_value(), hK_weight_change.get_value(), hK_weight_change.get_row(), hK_weight_change.get_column());
			cudaDeviceSynchronize();
			device_flow_compute << <blockPergrid, threadPerblock >> > (flow_gadient[round].get_value(), doutput_gate.get_value(), xO_weight_change.get_value(), xO_weight_change.get_row(), hO_weight_change.get_column());
			cudaDeviceSynchronize();
			device_flow_compute << <blockPergrid, threadPerblock >> > (flow_gadient[round].get_value(), dinput_gate.get_value(), xI_weight_change.get_value(), xI_weight_change.get_row(), hI_weight_change.get_column());
			cudaDeviceSynchronize();
			device_flow_compute << <blockPergrid, threadPerblock >> > (flow_gadient[round].get_value(), dfogot_gate.get_value(), xF_weight_change.get_value(), xF_weight_change.get_row(), hF_weight_change.get_column());
			cudaDeviceSynchronize();
			device_flow_compute << <blockPergrid, threadPerblock >> > (flow_gadient[round].get_value(), dK.get_value(), xK_weight_change.get_value(), xK_weight_change.get_row(), hK_weight_change.get_column());
			cudaDeviceSynchronize();

			Obias_change = Obias_change + doutput_gate * learning_rate;										// compute changing bias
			Ibias_change = Ibias_change + dinput_gate * learning_rate;
			Fbias_change = Fbias_change + dfogot_gate * learning_rate;
			Kbias_change = Kbias_change + dK * learning_rate;

			next_dc = mul_each(dc, fogot_gate);																	// compute next time step cell state error

			dh = next_dh;																						
			dc = next_dc;
			
			// try to descale exploding gadient
			double max_dh_value = std::max(get_max(dh), std::abs(get_min(dh)));									
			double max_dc_value = std::max(get_max(dc), std::abs(get_min(dc)));
			
			double flow_cap = std::sqrt(double(2) / v.size());
			if (max_dh_value > flow_cap) dh = dh * (flow_cap / max_dh_value);
			if (max_dc_value > flow_cap) dc = dc * (flow_cap / max_dc_value);
		}
													// compute initial cell state
		init_h_change = init_h_change + dh * learning_rate;
		init_c_change = init_c_change + dc * learning_rate;

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
		set_Matrix(xO_weight_change,number);
		set_Matrix(xF_weight_change,number);
		set_Matrix(xI_weight_change,number);
		set_Matrix(xK_weight_change,number);
		set_Matrix(hO_weight_change,number);
		set_Matrix(hF_weight_change,number);
		set_Matrix(hI_weight_change,number);
		set_Matrix(hK_weight_change,number);

		set_Matrix(Obias_change,number);
		set_Matrix(Fbias_change,number);
		set_Matrix(Ibias_change,number);
		set_Matrix(Kbias_change,number);

		set_Matrix(init_c_change,number);
		set_Matrix(init_h_change,number);
	}

	void mul_change_dependencies(const double& number) {
		xO_weight_change = xO_weight_change * number;
		xF_weight_change = xF_weight_change * number;
		xI_weight_change = xI_weight_change * number;
		xK_weight_change = xK_weight_change * number;
		hO_weight_change = hO_weight_change * number;
		hF_weight_change = hF_weight_change * number;
		hI_weight_change = hI_weight_change * number;
		hK_weight_change = hK_weight_change * number; 

		Obias_change = Obias_change * number;
		Fbias_change = Fbias_change * number;
		Ibias_change = Ibias_change * number;
		Kbias_change = Kbias_change * number;

		init_c_change = init_c_change * number;
		init_h_change = init_h_change * number;
	}



	void reconstruct(const std::size_t& size, const std::size_t& next,
		std::function<Matrix<double>(const Matrix<double>&)> _Iact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dIact_func = dtanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Fact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dFact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Oact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dOact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Kact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dKact_func = dtanh_func) {
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

	void reconstruct(const LayerId& set) {
		value.reconstruct(set.Layer_size, 1);
		Layer_type = Layer::LSTM;

		xO_weight.reconstruct(set.Layer_size, set.Layer_size);
		xF_weight.reconstruct(set.Layer_size, set.Layer_size);
		xI_weight.reconstruct(set.Layer_size, set.Layer_size);
		xK_weight.reconstruct(set.Layer_size, set.Layer_size);
		hO_weight.reconstruct(set.Layer_size, set.Layer_size);
		hF_weight.reconstruct(set.Layer_size, set.Layer_size);
		hI_weight.reconstruct(set.Layer_size, set.Layer_size);
		hK_weight.reconstruct(set.Layer_size, set.Layer_size);

		Obias.reconstruct(set.Layer_size, 1);
		Fbias.reconstruct(set.Layer_size, 1);
		Ibias.reconstruct(set.Layer_size, 1);
		Kbias.reconstruct(set.Layer_size, 1);

		xO_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xF_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xI_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xK_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hO_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hF_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hI_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hK_weight_change.reconstruct(set.Layer_size, set.Layer_size);

		Obias_change.reconstruct(set.Layer_size, 1);
		Fbias_change.reconstruct(set.Layer_size, 1);
		Ibias_change.reconstruct(set.Layer_size, 1);
		Kbias_change.reconstruct(set.Layer_size, 1);

		init_c.reconstruct(set.Layer_size, 1);
		init_h.reconstruct(set.Layer_size, 1);

		init_c_change.reconstruct(set.Layer_size, 1);
		init_h_change.reconstruct(set.Layer_size, 1);
		set_Matrix(init_c, 0);
		set_Matrix(init_h, 0);

		c.push_back(init_c);
		h.push_back(init_h);

		Iact_func = tanh_func;
		Fact_func = sigmoid_func;
		Oact_func = sigmoid_func;
		Kact_func = tanh_func;

		dIact_func = dtanh_func;
		dFact_func = dsigmoid_func;
		dOact_func = dsigmoid_func;
		dKact_func = dtanh_func;

		set_Layer(set.setting);
	}



	void rand_weight(const double& min,const double& max) {
		double* xO_weightHost = new double[value.get_size() * value.get_size()];
		double* xF_weightHost = new double[value.get_size() * value.get_size()];
		double* xI_weightHost = new double[value.get_size() * value.get_size()];
		double* xK_weightHost = new double[value.get_size() * value.get_size()];
		double* hO_weightHost = new double[value.get_size() * value.get_size()];
		double* hF_weightHost = new double[value.get_size() * value.get_size()];
		double* hI_weightHost = new double[value.get_size() * value.get_size()];
		double* hK_weightHost = new double[value.get_size() * value.get_size()];
		for (int i = 0; i < value.get_size() * value.get_size(); i++) {
			xO_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			xF_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			xI_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			xK_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			hO_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max) / 10;
			hF_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max) / 10;
			hI_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max) / 10;
			hK_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max) / 10;
		}
		cudaMemcpy(xO_weight.get_value(), xO_weightHost, xO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xI_weight.get_value(), xI_weightHost, xI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xF_weight.get_value(), xF_weightHost, xF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xK_weight.get_value(), xK_weightHost, xK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hO_weight.get_value(), hO_weightHost, hO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hI_weight.get_value(), hI_weightHost, hI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hF_weight.get_value(), hF_weightHost, hF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hK_weight.get_value(), hK_weightHost, hK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] xO_weightHost;
		delete[] xF_weightHost;
		delete[] xI_weightHost;
		delete[] xK_weightHost;
		delete[] hO_weightHost;
		delete[] hF_weightHost;
		delete[] hI_weightHost;
		delete[] hK_weightHost;
	}

	void rand_weight(std::pair<const double&, const double&> setting) {
		double* xO_weightHost = new double[value.get_size() * value.get_size()];
		double* xF_weightHost = new double[value.get_size() * value.get_size()];
		double* xI_weightHost = new double[value.get_size() * value.get_size()];
		double* xK_weightHost = new double[value.get_size() * value.get_size()];
		double* hO_weightHost = new double[value.get_size() * value.get_size()];
		double* hF_weightHost = new double[value.get_size() * value.get_size()];
		double* hI_weightHost = new double[value.get_size() * value.get_size()];
		double* hK_weightHost = new double[value.get_size() * value.get_size()];
		for (int i = 0; i < value.get_size() * value.get_size(); i++) {
			xO_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			xF_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			xI_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			xK_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			hO_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) / 10;
			hF_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) / 10;
			hI_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) / 10;
			hK_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) / 10;
		}
		cudaMemcpy(xO_weight.get_value(), xO_weightHost, xO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xI_weight.get_value(), xI_weightHost, xI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xF_weight.get_value(), xF_weightHost, xF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xK_weight.get_value(), xK_weightHost, xK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hO_weight.get_value(), hO_weightHost, hO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hI_weight.get_value(), hI_weightHost, hI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hF_weight.get_value(), hF_weightHost, hF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hK_weight.get_value(), hK_weightHost, hK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] xO_weightHost;
		delete[] xF_weightHost;
		delete[] xI_weightHost;
		delete[] xK_weightHost;
		delete[] hO_weightHost;
		delete[] hF_weightHost;
		delete[] hI_weightHost;
		delete[] hK_weightHost;
	}
	
	void rand_weight(std::function<double()> func) {
		double* xO_weightHost = new double[value.get_size() * value.get_size()];
		double* xF_weightHost = new double[value.get_size() * value.get_size()];
		double* xI_weightHost = new double[value.get_size() * value.get_size()];
		double* xK_weightHost = new double[value.get_size() * value.get_size()];
		double* hO_weightHost = new double[value.get_size() * value.get_size()];
		double* hF_weightHost = new double[value.get_size() * value.get_size()];
		double* hI_weightHost = new double[value.get_size() * value.get_size()];
		double* hK_weightHost = new double[value.get_size() * value.get_size()];
		for (int i = 0; i < value.get_size() * value.get_size(); i++) {
			xO_weightHost[i] = func();
			xF_weightHost[i] = func();
			xI_weightHost[i] = func();
			xK_weightHost[i] = func();
			hO_weightHost[i] = func() / 10;
			hF_weightHost[i] = func() / 10;
			hI_weightHost[i] = func() / 10;
			hK_weightHost[i] = func() / 10;
		}
		cudaMemcpy(xO_weight.get_value(), xO_weightHost, xO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xI_weight.get_value(), xI_weightHost, xI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xF_weight.get_value(), xF_weightHost, xF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xK_weight.get_value(), xK_weightHost, xK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hO_weight.get_value(), hO_weightHost, hO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hI_weight.get_value(), hI_weightHost, hI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hF_weight.get_value(), hF_weightHost, hF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hK_weight.get_value(), hK_weightHost, hK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] xO_weightHost;
		delete[] xF_weightHost;
		delete[] xI_weightHost;
		delete[] xK_weightHost;
		delete[] hO_weightHost;
		delete[] hF_weightHost;
		delete[] hI_weightHost;
		delete[] hK_weightHost;
	}

	void rand_weight(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		double* xO_weightHost = new double[value.get_size() * value.get_size()];
		double* xF_weightHost = new double[value.get_size() * value.get_size()];
		double* xI_weightHost = new double[value.get_size() * value.get_size()];
		double* xK_weightHost = new double[value.get_size() * value.get_size()];
		double* hO_weightHost = new double[value.get_size() * value.get_size()];
		double* hF_weightHost = new double[value.get_size() * value.get_size()];
		double* hI_weightHost = new double[value.get_size() * value.get_size()];
		double* hK_weightHost = new double[value.get_size() * value.get_size()];
		for (int i = 0; i < value.get_size() * value.get_size(); i++) {
			xO_weightHost[i] = func(value.get_row(), next);
			xF_weightHost[i] = func(value.get_row(), next);
			xI_weightHost[i] = func(value.get_row(), next);
			xK_weightHost[i] = func(value.get_row(), next);
			hO_weightHost[i] = func(value.get_row(), next) / 10;
			hF_weightHost[i] = func(value.get_row(), next) / 10;
			hI_weightHost[i] = func(value.get_row(), next) / 10;
			hK_weightHost[i] = func(value.get_row(), next) / 10;
		}
		cudaMemcpy(xO_weight.get_value(), xO_weightHost, xO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xI_weight.get_value(), xI_weightHost, xI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xF_weight.get_value(), xF_weightHost, xF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xK_weight.get_value(), xK_weightHost, xK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hO_weight.get_value(), hO_weightHost, hO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hI_weight.get_value(), hI_weightHost, hI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hF_weight.get_value(), hF_weightHost, hF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hK_weight.get_value(), hK_weightHost, hK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] xO_weightHost;
		delete[] xF_weightHost;
		delete[] xI_weightHost;
		delete[] xK_weightHost;
		delete[] hO_weightHost;
		delete[] hF_weightHost;
		delete[] hI_weightHost;
		delete[] hK_weightHost;
	}

	void rand_bias(const double& min, const double& max) {
		double* ObiasHost = new double[value.get_size()];
		double* FbiasHost = new double[value.get_size()];
		double* IbiasHost = new double[value.get_size()];
		double* KbiasHost = new double[value.get_size()];
		for (int i = 0; i < value.get_size(); i++) {
			ObiasHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			FbiasHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			IbiasHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			KbiasHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
		}
		cudaMemcpy(Obias.get_value(), ObiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Ibias.get_value(), IbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Fbias.get_value(), FbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Kbias.get_value(), KbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] ObiasHost;
		delete[] IbiasHost;
		delete[] FbiasHost;
		delete[] KbiasHost;
	}

	void rand_bias(std::pair<const double&,const double&> setting) {
		double* ObiasHost = new double[value.get_size()];
		double* FbiasHost = new double[value.get_size()];
		double* IbiasHost = new double[value.get_size()];
		double* KbiasHost = new double[value.get_size()];
		for (int i = 0; i < value.get_size(); i++) {
			ObiasHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			FbiasHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			IbiasHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			KbiasHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		}
		cudaMemcpy(Obias.get_value(), ObiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Ibias.get_value(), IbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Fbias.get_value(), FbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Kbias.get_value(), KbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] ObiasHost;
		delete[] IbiasHost;
		delete[] FbiasHost;
		delete[] KbiasHost;
	}
	
	void rand_bias(std::function<double()> func) {
		double* ObiasHost = new double[value.get_size()];
		double* FbiasHost = new double[value.get_size()];
		double* IbiasHost = new double[value.get_size()];
		double* KbiasHost = new double[value.get_size()];
		for (int i = 0; i < value.get_size(); i++) {
			ObiasHost[i] = func();
			FbiasHost[i] = func();
			IbiasHost[i] = func();
			KbiasHost[i] = func();;
		}
		cudaMemcpy(Obias.get_value(), ObiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Ibias.get_value(), IbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Fbias.get_value(), FbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Kbias.get_value(), KbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] ObiasHost;
		delete[] IbiasHost;
		delete[] FbiasHost;
		delete[] KbiasHost;
	}

	void rand_bias(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		double* ObiasHost = new double[value.get_size()];
		double* FbiasHost = new double[value.get_size()];
		double* IbiasHost = new double[value.get_size()];
		double* KbiasHost = new double[value.get_size()];
		for (int i = 0; i < value.get_size(); i++) {
			ObiasHost[i] = func(value.get_row(), next);
			FbiasHost[i] = func(value.get_row(), next);
			IbiasHost[i] = func(value.get_row(), next);
			KbiasHost[i] = func(value.get_row(), next);
		}
		cudaMemcpy(Obias.get_value(), ObiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Ibias.get_value(), IbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Fbias.get_value(), FbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Kbias.get_value(), KbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] ObiasHost;
		delete[] IbiasHost;
		delete[] FbiasHost;
		delete[] KbiasHost;
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

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dOact_func() {
		return dOact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dFact_func() {
		return dFact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dIact_func() {
		return dIact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dKact_func() {
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
		Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h.back()) + Ibias);
		Matrix<double> fogot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);
		Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);
		Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);
		std::cout << "--------------LSTM Layer----------\n\n";
		value.print();
		std::cout << "--------input--------\n";
		input_gate.print();
		std::cout << "--------fogot--------\n";
		fogot_gate.print();
		std::cout << "--------output-------\n";
		output_gate.print();
		std::cout << "----------K----------\n";
		K.print();
	}
protected:
	void print_xO_weight() {
		std::cout << "  -----x-output weight----\n";
		xO_weight.print();
	}

	void print_xF_weight() {
		std::cout << "  -----x-fogot weight----\n";
		xF_weight.print();
	}

	void print_xI_weight() {
		std::cout << "  -----x-input weight----\n";
		xI_weight.print();
	}

	void print_xK_weight() {
		std::cout << "  -----x-k    weight----\n";
		xK_weight.print();
	}

	void print_hO_weight() {
		std::cout << "  -----h-output weight----\n";
		hO_weight.print();
	}

	void print_hF_weight() {
		std::cout << "  -----h-fogot weight----\n";
		hF_weight.print();
	}

	void print_hI_weight() {
		std::cout << "  -----h-input weight----\n";
		hI_weight.print();
	}

	void print_hK_weight() {
		std::cout << "  -----h-K     weight----\n";
		hK_weight.print();
	}

	void print_Obias() {
		std::cout << "   ---output bias------\n";
		Obias.print();
	}

	void print_Fbias() {
		std::cout << "   ---fogot bias------\n";
		Fbias.print();
	}

	void print_Ibias() {
		std::cout << "   ---input bias------\n";
		Ibias.print();
	}

	void print_Kbias() {
		std::cout << "   ---K     bias------\n";
		Kbias.print();
	}

	void print_init() {
		std::cout << " -------- init----------\n";
		init_c.print();

		init_h.print();
	}



	void set_Layer(const std::string& setting) {
		int size = setting.size();
		int i = 0;
		std::string a;
		while (i < size) {
			a = get_text(setting, i);
			if (a == "Iact")
				universal_set_func(Iact_func, setting, i);
			else if (a == "Fact")
				universal_set_func(Fact_func, setting, i);
			else if (a == "Oact")
				universal_set_func(Oact_func, setting, i);
			else if (a == "Kact")
				universal_set_func(Kact_func, setting, i);
			else if (a == "dIact")
				universal_set_func(dIact_func, setting, i);
			else if (a == "dFact")
				universal_set_func(dFact_func, setting, i);
			else if (a == "dOact")
				universal_set_func(dOact_func, setting, i);
			else if (a == "dKact")
				universal_set_func(dKact_func, setting, i);
			else if (a == "learning_rate")
				set_learning_rate(setting, i);
			else if (a == "")
				;
			else throw "command not found";
		}
	}

	void set_learning_rate(const std::string& str, int& i) {
		double a = get_number(str, i);
		learning_rate = a;
	}


	Matrix<double> xO_weight;																				// weight for input -> output gate
	Matrix<double> xF_weight;																				// weight for input -> forgot gate
	Matrix<double> xI_weight;																				// weight for input -> input gate
	Matrix<double> xK_weight;																				// weight for input -> K
	Matrix<double> hO_weight;																				// weight for hidden -> output gate
	Matrix<double> hF_weight;																				// weight for hidden -> forgot gate
	Matrix<double> hI_weight;																				// weight for hidden -> input gate
	Matrix<double> hK_weight;																				// weight for hidden -> K

	Matrix<double> Obias;																					// bias for output gate
	Matrix<double> Fbias;																					// bias for forgot gate	
	Matrix<double> Ibias;																					// bias for input gate
	Matrix<double> Kbias;																					// bias for K

	Matrix<double> init_c;																					// initial cell state
	Matrix<double> init_h;																					// initial hidden

	Matrix<double> xO_weight_change;																		// changing weight for input -> output gate
	Matrix<double> xF_weight_change;																		// changing weight for input -> forgot gate
	Matrix<double> xI_weight_change;																		// changing weight for input -> input gate
	Matrix<double> xK_weight_change;																		// changing weight for input -> K
	Matrix<double> hO_weight_change;																		// changing hidden for input -> output gate
	Matrix<double> hF_weight_change;																		// changing hidden for input -> forgot gate
	Matrix<double> hI_weight_change;																		// changing hidden for input -> input gate
	Matrix<double> hK_weight_change;																		// changing hidden for input -> K

	Matrix<double> Obias_change;																			// changing bias for output gate																	
	Matrix<double> Fbias_change;																			// changing bias for forgot gate
	Matrix<double> Ibias_change;																			// changing bias for input gate
	Matrix<double> Kbias_change;																			// changing bias for K

	Matrix<double> init_c_change;																			// changing initial cell state
	Matrix<double> init_h_change;																			// changing initial hidden
	
	std::vector<Matrix<double>> c;																			// cell state memory
	std::vector<Matrix<double>> h;																			// output memory

	std::function<Matrix<double>(const Matrix<double>&)> Oact_func = sigmoid_func;							// activate function for output gate
	std::function<Matrix<double>(const Matrix<double>&)> Fact_func = sigmoid_func;							// activate function for forgot gate
	std::function<Matrix<double>(const Matrix<double>&)> Iact_func = tanh_func;								// activate function for input gate
	std::function<Matrix<double>(const Matrix<double>&)> Kact_func = tanh_func;								// activate function for K

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dOact_func = dsigmoid_func;	// derivative of activate function for output gate
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dFact_func = dsigmoid_func;	// derivative of activate function for forgot gate
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dIact_func = dtanh_func;	// derivative of activate function for input gate
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dKact_func = dtanh_func;	// derivative of activate function for K
};