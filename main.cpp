#include "Header.h"
#include "Neural_Network.cpp"
void sight_test(Neural_Network& AI) {
	AI.print_weight();
	AI.print_bias();
	std::cin.get();
}
double sigmoid(const double& input) {
	double reuslt = double(1) / (double(1) + std::exp(-input));
	if (reuslt == reuslt)
		return reuslt;
	return 0.00001;
}
double inv_sigmoid(const double& input) {
	double result = -std::log(double(1) / input - 1);
	if (result != result) {
		return -60;
	}
	return result;
}
const char Model_file_name[] = "file/Model.txt";
const char Data_file_name[] = "file/Data.txt";
const char output_file_name[] = "file/Output.txt";
std::vector<LayerId> get_Model_layerId();
std::vector<Matrix<double>> get_data(const std::size_t& input_size);
std::vector<std::pair<double, double>> rand_weight_setting;
std::vector<std::pair<double, double>> rand_bias_setting;
int data_range;
int learn_range = 10000;
int test_range;

int input_range = 50;
int output_range = 1;

double max_data = 0;
double min_data = 100000000;
double max_value;
void print_output(const double& number) {
	static std::ofstream Output_file(output_file_name);
	static double last_value = 0;
	last_value += number;
	while (!Output_file.is_open()) {

	}
	Output_file << number << std::endl;
}
void learn(Neural_Network& AI,const std::vector<Matrix<double>> Data, int start) {
	static int round_passed = 0;
	round_passed++;
	if(round_passed % 10 == 0)
		std::cout << "AI has beened train : " << (double(round_passed) / learn_range) * 100 << "%\n";
	for (int i = start; i < start + input_range; i++) {
		AI.feedforward(Data[i]);
	}
	print_output(mapping(inv_sigmoid(AI.get_output()[0][0]),-100 ,100 , -max_value,max_value));
	AI.set_change_dependencies(0);
	//sight_test(AI);

	for (int i = start + input_range; i < start + input_range + output_range; i++) {
		AI.backpropagation(Data[i]);
	}
	AI.change_dependencies();
	AI.fogot_all();

	//sight_test(AI);
}
Matrix<double> predict(Neural_Network& AI,const std::vector<Matrix<double>>& Data,int start) {
	for (int i = start; i < start + input_range; i++) {
		AI.feedforward(Data[i]);
	}
	AI.fogot_all();
	print_output(mapping(inv_sigmoid(AI.get_output()[0][0]), -100, 100, -max_value, max_value));
	return AI.get_output();
}
int main() {
	std::srand(std::time(0));
	std::vector<LayerId> Model = get_Model_layerId(); std::cout << "Model load success\n";
	std::vector<Matrix<double>> Data = get_data(Model[0].Layer_size); std::cout << "Data load success\n";
	Neural_Network AI(Model);
	AI.set_all_learning_rate(0.001);
	
	// set rand_setting
	for (int i = 0; i < AI.get_layer_size() - 1; i++) {
		const double lopk = double(1) / (double(1) + Model[i].Layer_size);
		rand_weight_setting.push_back({-lopk,lopk});
		rand_bias_setting.push_back({-lopk,lopk});
	}
	AI.rand_weight(rand_weight_setting);
	AI.rand_bias(rand_bias_setting);
	for (int i = Data.size() - 1; i > 0; i--) {
		Data[i] = Data[i] - Data[i - 1];
	}
	Data[0][0][0] = 0;
	
	for (int i = 0; i < Data.size(); i++) {
		max_data = std::max(max_data, Data[i][0][0]);
		min_data = std::min(min_data, Data[i][0][0]);
	}
	max_value = std::max(max_data, std::abs(min_data));
	for (int i = 0; i < Data.size(); i++) {
		Data[i][0][0] = sigmoid(mapping(Data[i][0][0], -max_value, max_value, -100, 100));
	}

	sight_test(AI);
	for (int i = 0; i < learn_range - input_range - output_range; i++) {
		learn(AI, Data, i);
	}
	sight_test(AI);

	for (int i = learn_range - input_range - output_range; i < Data.size() - input_range - output_range; i++) {
		Data[i + input_range] = predict(AI,Data,i);
	}
	std::cout << "Everything seems to be fine";

	static int vol = 0;
	while (true) {
		double get_input;
		Matrix<double> input(1,1);
		std::cin >> get_input;
		input[0][0] = get_input;
		AI.feedforward(input);
		std::cout << AI.get_output()[0][0];
		++vol;
		if (vol >= 7)
			AI.fogot(1);
		AI.print_value();
	}
	return 0;
}

std::vector<LayerId> get_Model_layerId() {
	std::vector<LayerId> result;
	std::ifstream Model_file(Model_file_name);
	while (!Model_file.is_open()) {

	}
	while (!Model_file.eof()) {
		std::size_t _size;
		int _layer_type;
		Model_file >> _layer_type >> _size;
		result.push_back(LayerId(Layer::type(_layer_type),_size));
	}
	return result;
}

std::vector<Matrix<double>> get_data(const std::size_t& input_size) {
	std::vector<Matrix<double>> _Data;
	std::ifstream Data_file(Data_file_name);
	data_range = 0;
	while (!Data_file.is_open()) {

	}
	std::cout << "Data file opened \n";
	while (!Data_file.eof()) {
		_Data.push_back(Matrix<double>(input_size, 1));
		for (int i = 0; i < input_size; i++) {
			Data_file >> _Data.back()[i][0];
		}
		++data_range;
	}
	test_range = data_range - learn_range;
	if (test_range <= 0)
		throw "invalid learn-range";
	return _Data;
}