// Wheel-NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <Eigen/Dense>
#include "NeuralNetwork.h"
#include "CSVParser.h"
#define PRINT(x) std::cout << x << std::endl
int main()
{
	DataPreprocessing::CSVParser data_parser("data.csv");
	auto data = data_parser.GetParsedResult();

	//Optimal layers for XOR problem: 2 3 2

	std::vector<Layer> layers;
	layers.emplace_back(2);
	layers.emplace_back(4);
	//layers.emplace_back(2);

	layers.emplace_back(1);

	//std::vector<float> inputs{1,1};
	//std::vector<float> target{ 0 };
	NeuralNetwork nn(layers);
	srand((unsigned int)time(NULL));
	for (int i = 0; i < 50000; i++) {
		std::vector<float> inputs;
		std::vector<float> target;
		int x2 = ((double)rand() / (RAND_MAX)) > 0.5 ? 1 : 0;
		int x1 = ((double)rand() / (RAND_MAX)) > 0.5 ? 1 : 0;
		
		int y = 1;

		//XOR
		if (x1 == x2)
			y = 1;
		else
		{
			y = 0;
		}

		////OR
		//if (x1 == 1 || x2 == 1)
		//	y = 1;
		//else
		//{
		//	y = 0;
		//}

		////AND
		//if (x1 == 1 && x2 == 1)
		//	y = 1;
		//else
		//{
		//	y = 0;
		//}

		/*PRINT("Inputs:");
		PRINT(x1);
		PRINT(x2);
		PRINT("LABEL");
		PRINT(y);*/
		inputs.emplace_back(x1);
		inputs.emplace_back(x2);

		target.emplace_back(y);

		nn.SetInput(inputs);
		nn.Train(inputs, target);
	}

	/*nn.SetInput(inputs);
	nn.Train(inputs, target);*/

	/*for (int j = 0; j < 300; j++) {
		for (int i = 1; i < data.size(); ++i)
		{
			std::vector<float> inputs;
			std::vector<float> target;
			inputs.emplace_back(std::atof(data[i][0].c_str()));
			inputs.emplace_back(std::atof(data[i][1].c_str()));

			int label = std::atoi(data[i][2].c_str());
			target.emplace_back(label);
			std::cout << "training... " << std::endl;
			nn.SetInput(inputs);
			nn.Train(inputs, target);

		}
	}*/

	std::vector<std::vector<float>> all_data_to_predict;
	all_data_to_predict.emplace_back(std::vector<float>{ 0, 1 });
	all_data_to_predict.emplace_back(std::vector<float>{ 1, 0 });
	all_data_to_predict.emplace_back(std::vector<float>{ 1, 1 });
	all_data_to_predict.emplace_back(std::vector<float>{ 0, 0 });

	for (int i = 0; i < all_data_to_predict.size(); ++i)
	{
		std::cout << "prediction: " << nn.predict(all_data_to_predict[i])[0] << std::endl;
	}
}
