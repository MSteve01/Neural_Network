#pragma once
#include "Header.h"
#include "Matrix.cpp"

std::function<Matrix<double>(const Matrix<double>&)> sigmoid_func = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = double(1) / (double(1) + std::exp(-input[i][j]));
			if (result[i][j] != result[i][j]) {
				result[i][j] = 0.000001;
			}
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> dsigmoid_func = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = std::exp(-input[i][j]) / std::pow(double(1) + std::exp(-input[i][j]), 2.0);
			if (result[i][j] != result[i][j])
				result[i][j] = 0.000001;
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> tanh_func = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = std::tanh(input[i][j]);
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> dtanh_func = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = double(1) - std::pow(std::tanh(input[i][j]), 2.0);
		}
	}
	return result;
};

double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2) {
	return ((value - min1) / (max1 - min1) * (max2 - min2)) + min2;
}

Matrix<double> mul_each(const Matrix<double>& left, const Matrix<double>& right) {
	if (left.get_row() != right.get_row() || left.get_column() != right.get_column())
		throw "invalid multiply each elemenet";
	Matrix<double> result(left.get_row(),left.get_column());
	for (int i = 0; i < left.get_row(); i++) {
		for (int j = 0; j < left.get_column(); j++) {
			result[i][j] = left[i][j] * right[i][j];
		}
	}
	return result;
}

void set_Matrix(Matrix<double>& M, double value) {
	for (int i = 0; i < M.get_row(); i++) {
		for (int j = 0; j < M.get_column(); j++) {
			M[i][j] = value;
		}
	}
}