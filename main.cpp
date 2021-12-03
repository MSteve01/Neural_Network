#include "Header.h"
#include "Neural_Network.cpp"
#include <thread>
#include <chrono>
#include <windows.h>
const char* Model_file_name = "file/Model.txt";
const char* DataBase_file_name = "file/Data.txt";
const char* output_file_name = "file/Output.txt";
const char* RandWeightSetting_file_name = "file/Rand_weight_setting.txt";
const char* RandBiasSetting_file_name = "file/Rand_bias_setting.txt";
const char* Lost_file_name = "file/lost.txt";
int data_range;
int learning_range;
int testing_range;
int input_range;
int output_range;
int have_trained = 0;
double train_speed = 1;
int load_range = 0;
void print_time(long long int input) {
	long long int second = input / 1000;
	long long int minute = (second - (second % 60)) / 60;
	long long int hour = (minute - (minute % 60)) / 60;
	minute %= 60;
	second %= 60;
	std::cout << hour << " hours " << minute << " minutes " << second << " seconds\n";
}
std::function<double()> random_func = []() {
	//return std::pow(-1, rand() % 2) * double(rand() % 20000) / 20000;
	return std::pow(-1, rand() % 2) * std::tanh(double(rand() % 30000) / 10000);
};
std::function<double(std::size_t, std::size_t)> random_func2 = [](std::size_t size, std::size_t next) {
	return std::pow(-1, rand() % 2) * (double(rand() % 20000) / 10000) * std::sqrt(double(2) / size);
};
std::function<double()> zero = []() {
	return 0;
};

void print_process() {
	while (have_trained + input_range + output_range < learning_range) {
		std::cout << train_speed << '\n';
		std::system("CLS");
		std::cout << "AI has been trained : " << (double(have_trained) / learning_range) * 100 << "%\n";
		long long int remaining = learning_range - have_trained;
		std::cout << "estimated traingin time : ";
		print_time(train_speed * remaining);
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
}

void print_number(int& b,int max) {
	while (b < max) {
		std::cout << b << " \ " << max << "\n";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
}

double compute_lost(Matrix<double> input, Matrix<double> target) {
	double result = 0;
	for (int i = 0; i < input.get_row(); i++) {
		result += std::pow((target[i][0] - input[i][0]), 2.0);
	}
	return result;
}

std::vector<LayerId> load_model() {
	std::vector<LayerId> Model;
	std::ifstream Model_file(Model_file_name);
	while (!Model_file.eof()) {
		int input1, input2; Model_file >> input1 >> input2;
		Model.push_back(LayerId(Layer::type(input1), input2));
	}
	Model_file.close();
	return Model;
}

std::vector<Matrix<double>> load_data(std::size_t input_size) {
	std::vector<Matrix<double>> Data;
	std::ifstream DataBase_file(DataBase_file_name);
	while (!DataBase_file.eof()) {
		Data.push_back(Matrix<double>(input_size,1));
		for (int i = 0; i < input_size; i++) {
			DataBase_file >> Data.back()[i][0];
		}
	}
	return Data;
}std::vector<Matrix<double>> load_data(std::size_t input_size, std::size_t data_range) {
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

void normalization(std::vector<Matrix<double>>& Data) {
	for (int i = 0; i < Data[0].get_row(); i++) {
		double min = Data[0][i][0];
		double max = Data[0][i][0];
		for (int j = 0; j < Data.size(); j++) {
			if (Data[j][i][0] < min) min = Data[j][i][0];
			if (Data[j][i][0] > max) max = Data[j][i][0];
		}

		for (int j = 0; j < Data.size(); j++) {
			Data[j][i][0] = mapping(Data[j][i][0], min, max, 0, 1);
		}
	}
}

double learn(Neural_Network& AI, std::vector<Matrix<double>> Data, int start) {
	auto start_time = std::chrono::system_clock::now();

	double lost = 0;
	for (int i = start; i < start + input_range; i++) {
		AI.feedforward(Data[i]);
	}

	AI.mul_change_dependencies(4);
	for (int i = start + input_range; i < start + input_range + output_range; i++) {
		AI.backpropagation(Data[i]);
		lost += compute_lost(AI.get_output(), Data[i]);
	}
	AI.mul_change_dependencies(0.2);

	have_trained++;
	AI.change_dependencies();
	AI.fogot_all();
	//std::cout << lost << std::endl;

	auto end_time = std::chrono::system_clock::now();
	std::chrono::duration<double> d = end_time - start_time;
	train_speed = d.count() * 1000;
	return std::sqrt(lost);
}

std::vector<Matrix<double>> predict(Neural_Network& AI, std::vector<Matrix<double>> Data, int start) {
	std::vector<Matrix<double>> result;
	for (int i = start; i < start + input_range; i++) {
		AI.feedforward(Data[i]);
	}

	for (int i = start + input_range; i < start + input_range + output_range; i++) {
		result.push_back(Matrix<double>(AI.get_input_size(), 1)); result.back() = AI.get_output();
		AI.feedforward(result.back());
	}

	AI.fogot_all();
	return result;
}

int main() {
	try {
		std::srand(std::time(0));
		Neural_Network AI(load_model()); std::cout << "Model was leaded successfully\n";
		std::vector<Matrix<double>> Data; // = load_data(AI.get_input_size()); std::cout << "Data was loaded successfully\n";
		//data_range = Data.size(); std::cout << "Data range : " << data_range << "\nlearning range : ";
		std::cout << "data range : "; std::cin >> data_range; std::cout << "load_range : "; std::cin >> load_range; std::cout << "learing range : ";
		std::cin >> learning_range;
		testing_range = data_range - learning_range;
		//normalization(Data); std::cout << "Data was normalizetion sucessfully\n";
		std::cout << "input range : "; std::cin >> input_range;
		std::cout << "output range : "; std::cin >> output_range;
		std::vector<std::function<double(std::size_t, std::size_t)>> Weight_setting; for (int i = 0; i < AI.get_layer_size() - 1; i++) { Weight_setting.push_back(random_func2); }
		std::cout << "Load rand weigth setting successfully\n";
		std::vector<std::function<double()>> Bias_setting; for (int i = 0; i < AI.get_layer_size() - 1; i++) { Bias_setting.push_back(zero); }
		std::cout << "Load rand bias setting successfully\n";

		if (Weight_setting.size() < AI.get_layer_size() - 1) {
			std::cout << "Weight setting doesn't match AIsize\n"; return 0;
		}
		if (Bias_setting.size() < AI.get_layer_size() - 1) {
			std::cout << "Bias setting doesn't maych AIsize\n"; return 0;
		}
		AI.set_all_learning_rate(0.0001);
		AI.rand_weight(Weight_setting);
		AI.rand_bias(Bias_setting);
		AI.set_all_drop_out_rate(0.1);
		AI.set_change_dependencies(0);
		//	std::thread s1(print_process);
		std::ofstream output_file(output_file_name);
		std::ofstream lost_file(Lost_file_name);
		for (int i = 0; i + input_range + output_range < learning_range; i++) {
			if (i % load_range == 0) {
				Data = load_data(AI.get_input_size(), load_range);
			}
			int pos = rand() % (/*learning_range - input_range - output_range*/ load_range - input_range - output_range);
			lost_file << learn(AI, Data, pos) << "\n";
			/*while (true) {
				Matrix<double> input(1,1);
				std::cin >> input[0][0];
				if (input[0][0] != -2 && input[0][0] != -3) {
					AI.feedforward(input);
					AI.print_value();
				}
				else if (input[0][0] == -2) {
					AI.fogot_all();
				}
				else
					break;
			}
			std::system("CLS");
			*/
			//output_file << AI.get_output()[0][0] << std::endl;
			int spos;
			double smax = -10000000;
			for (int i = 0; i < 256; i++) {
				if (smax < AI.get_output()[i][0]) {
					spos = i;
					smax = AI.get_output()[i][0];
				}
			}


			std::cout << "     " << char(spos);
			double max_data = -10000; int super_pos;
			if (i % 30 == 0) {
				//system("CLS");
				for (int i = 0; i < 256; i++) {
					if (i == spos)
						SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 240);
					else if (AI.get_output()[i][0] < 0.00001)
						SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 8);
					std::cout << AI.get_output()[i][0]; if (i > 20) std::cout << '(' << char(i) << ')';
					SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 15);
					std::cout << " \t";
					if ((i % 8) == 0)
						std::cout << "\n";
				}
			}

			for (int q = 0; q < 256; q++) {
				if (Data[pos][q][0]) {
					std::cout << char(q) << std::endl;
					break;
				}
			}
		}
		std::cout << "started testing\n";
		Data = load_data(AI.get_input_size(), input_range);
		for (int i = 0; i + input_range + output_range < testing_range; i++) {
			predict(AI, Data, i);
			Matrix<double> output(256, 1);
			output = AI.get_output();
			double max = -1, pos;
			Data.push_back(Matrix<double>(AI.get_input_size(), 1));
			for (int j = 0; j < 256; j++) {
				if (output[j][0] > max) {
					max = output[j][0];
					pos = j;
				}
				Data.back()[j][0] = 0;
			}
			output_file << char(pos);
			Data.back()[pos][0] = 1;
		}
		//	s1.join();
		return 0;
	}
	catch (std::string Error) {
		std::cout << Error << std::endl;
	}
	std::cin.get();
}