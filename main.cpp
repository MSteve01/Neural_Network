#include "Header.h"
#include "Neural_Network.cpp"
#include <thread>
#include <chrono>
#include <windows.h>

// file name
const char* Model_file_name = "file/Model.txt";
const char* DataBase_file_name = "file/Data.txt";
const char* output_file_name = "file/Output.txt";
const char* RandWeightSetting_file_name = "file/Rand_weight_setting.txt";
const char* RandBiasSetting_file_name = "file/Rand_bias_setting.txt";
const char* Lost_file_name = "file/lost.txt";

// variable
int data_range;
int learning_range;
int testing_range;
int input_range;
int output_range;
int have_trained = 0;
double train_speed = 1;
int load_range = 0;

// import functions

extern std::function<Matrix<double>(const Matrix<double>&)> soft_max;
extern std::function<Matrix<double>(const Matrix<double>&)> linear_func;
extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dsoft_max;
extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dlinear_func;
extern std::function<double(const Matrix<double>&, const Matrix<double>&)> catagorical_CEnt_loss_func;
extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dcatagorical_CEnt_loss_func;

// weight and bias initialization function
std::function<double()>
random_func = []() {
	//return std::pow(-1, rand() % 2) * double(rand() % 20000) / 20000;
	return std::pow(-1, rand() % 2) * std::tanh(double(rand() % 30000) / 10000);
};
std::function<double(std::size_t, std::size_t)> 
random_func2 = [](std::size_t size, std::size_t next) {
	return std::pow(-1, rand() % 2) * (double(rand() % 20000) / 10000) * std::sqrt(double(2) / size);
};
std::function<double()> 
zero = []() {
	return 0;
};


std::vector<LayerId> load_model() {
	std::vector<LayerId> Model;
	std::ifstream Model_file(Model_file_name);
	while (!Model_file.eof()) {
		int input1, input2; Model_file >> input1 >> input2;
		std::string setting; std::getline(Model_file, setting);
		Model.push_back(LayerId(Layer::type(input1), input2, setting));
	}
	Model_file.close();
	return Model;
}

std::vector<Matrix<double>> load_data(std::size_t input_size) { // loas the whole data
	std::vector<Matrix<double>> Data;
	std::ifstream DataBase_file(DataBase_file_name);
	while (!DataBase_file.eof()) {
		Data.push_back(Matrix<double>(input_size,1));
		for (int i = 0; i < input_size; i++) {
			DataBase_file >> Data.back()[i][0];
		}
	}
	return Data;
}

std::vector<Matrix<double>> load_data(std::size_t input_size, std::size_t data_range) { // load the data in specific range
	int loop = 0;
	std::vector<Matrix<double>> Data;
	static std::ifstream DataBase_file(DataBase_file_name);
	while (!DataBase_file.eof() && loop < data_range) {
		Data.push_back(Matrix<double>(input_size, 1));
		for (int i = 0; i < input_size; i++) {
			DataBase_file >> Data.back()[i][0];
		}
		loop++;
	}
	//DataBase_file.close();
	return Data;
}

std::vector<std::pair<double, double>> load_rand_weight_setting() {
	std::vector<std::pair<double, double>> setting;
	std::ifstream RandWeightFile(RandWeightSetting_file_name);
	while (!RandWeightFile.eof()) {
		double input1, input2;
		RandWeightFile >> input1 >> input2;
		setting.push_back({ input1, input2 });
	}
	RandWeightFile.close();
	return setting;
}

std::vector<std::pair<double, double>> load_rand_bias_setting() {
	std::vector<std::pair<double, double>> setting;
	std::ifstream RandBiasFile(RandBiasSetting_file_name);
	while (!RandBiasFile.eof()) {
		double input1, input2;
		RandBiasFile >> input1 >> input2;
		setting.push_back({ input1,input2 });
	}
	RandBiasFile.close();
	return setting;
}



double learn(Neural_Network& AI, std::vector<Matrix<double>> Data, int start) {
	double lost = 0;
	for (int i = start; i < start + input_range; i++) {															// feedforward
		AI.feedforward(Data[i]);
	}

	AI.mul_change_dependencies(0);
	for (int i = start + input_range; i < start + input_range + output_range; i++) {							// backpropagation
		AI.backpropagation(Data[i]);
		lost += AI.get_loss(Data[i]);
	}

	have_trained++;
	AI.change_dependencies();
	AI.fogot_all();

	return lost;
}

std::vector<Matrix<double>> predict(Neural_Network& AI, std::vector<Matrix<double>> Data, int start) {
	std::vector<Matrix<double>> result;
	for (int i = start; i < start + input_range; i++) {
		AI.feedforward(Data[i]);
	}

	for (int i = start + input_range; i < start + input_range + output_range; i++) {
		result.push_back(Matrix<double>(AI.get_input_size(), 1)); result.back() = AI.get_output();
		//AI.feedforward(result.back());
	}

	AI.fogot_all();
	return result;
}

int main() {
	try {
		std::srand(std::time(0));
		Neural_Network AI(load_model(), catagorical_CEnt_loss_func, dcatagorical_CEnt_loss_func); 
		std::cout << "Model was leaded successfully\n";
		
		std::vector<Matrix<double>> Data;
		std::cout << "data range : "; std::cin >> data_range;													// get setting 
		std::cout << "load_range : "; std::cin >> load_range; 
		std::cout << "learing range : ";std::cin >> learning_range;	
		std::cout << "input range : "; std::cin >> input_range;
		std::cout << "output range : "; std::cin >> output_range;
		
		testing_range = data_range - learning_range;
		
		std::vector<std::function<double(std::size_t, std::size_t)>> Weight_setting; 
		for (int i = 0; i < AI.get_layer_size() - 1; i++) { Weight_setting.push_back(random_func2); }
		std::cout << "Load rand weigth setting successfully\n";
		std::vector<std::function<double()>> Bias_setting;
		for (int i = 0; i < AI.get_layer_size() - 1; i++) { Bias_setting.push_back(zero); }
		std::cout << "Load rand bias setting successfully\n";

		if (Weight_setting.size() < AI.get_layer_size() - 1) {													// check for error setting
			std::cout << "Weight setting doesn't match AIsize\n"; return 0;
		}
		if (Bias_setting.size() < AI.get_layer_size() - 1) {
			std::cout << "Bias setting doesn't maych AIsize\n"; return 0;
		}



		AI.set_all_learning_rate(0.001);																		// set up AI
		AI.rand_weight(Weight_setting);
		AI.rand_bias(Bias_setting);
		AI.set_change_dependencies(0);

		std::ofstream output_file(output_file_name);
		std::ofstream lost_file(Lost_file_name);

		for (int i = 0; i + input_range + output_range < learning_range; i++) {									// loop though every data for learning
			
			if (i % load_range == 0) {																			// loas data
				Data = load_data(AI.get_input_size(), load_range);
			}

			int pos = rand() % ( load_range - input_range - output_range);										// random pattern for training

			lost_file << learn(AI, Data, pos) << "\n";															// learn and put lost into the file

			int spos;																							// print result and answer of learning
			for (int j = 0; j < output_range; j++) {
				double smax = -10000000;
				for (int i = 0; i < 256; i++) {
					if (smax < AI.get_output()[i][0]) {
						spos = i;
						smax = AI.get_output()[i][0];
					}
				}


				std::cout << " " << char(spos);
				double max_data = -10000; int super_pos;
			}
		}
		std::cout << "started testing\n";																		// predict
		Data = load_data(AI.get_input_size(), input_range);
		for (int i = 0; i + input_range + output_range < testing_range; i+=output_range) {
			std::vector<Matrix<double>> output = predict(AI, Data, i);
			for (int k = 0; k < output_range; k++) {
				Data.push_back(output[k]);
				double pos = 0;
				double max = 1000000;
				for (int j = 0; j < 256; j++) {
					if (output[k][j][0] > max) {
						max = output[k][j][0];
						pos = j;
					}
					Data.back()[j][0] = 0;
				}
				output_file << char(pos);
				Data.back()[pos][0] = 1;
			}
		}
		//	s1.join();
		return 0;
	}
	catch (std::string Error) {																					// catch the error
		std::cout << Error << std::endl;
		std::cin.get();
		return 0;
	}
	std::cin.get();
}