#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers) : m_layers(layers)
{
	srand((unsigned int)time(NULL));
	InitWeights();
}

void NeuralNetwork::SetInput(std::vector<float> inputs)
{
	//Don't set the last input since it's the bias
	if (inputs.size() != m_layers[0].GetInputs()->size())
	{
		std::cout << "Error: size of input doesn't match size of the first layer" << std::endl;
		return;
	}

	for (int i = 0; i < m_layers[0].GetInputs()->size(); ++i)
	{
		(*(m_layers[0].GetInputs()))[i] = inputs[i];
	}
}

std::vector<float> NeuralNetwork::predict(std::vector<float> inputs)
{
	SetInput(inputs);
	Eigen::MatrixXf prev_output;
	for (int i = 0; i < m_layers.size(); ++i)
	{
		if (i < m_layers.size() - 1)
		{
			int row_num = m_layers[i].GetInputs()->size();
			int col_num = m_layers[i + 1].GetInputs()->size();
			

			Eigen::MatrixXf current_input(row_num, 1);

			Eigen::MatrixXf weight_mat(col_num, row_num);
			weight_mat.setOnes();

			Eigen::MatrixXf current_output(row_num, 1);

			if (i == 0)
			{
				float* input_array_raw = m_layers[i].GetInputs()->data();
				current_input = Eigen::Map<Eigen::MatrixXf>(input_array_raw, row_num, 1);
			}
			else
			{
				current_input = prev_output;
			}
			current_output = m_weights[i] * current_input;

			//Add the bias
			current_output = current_output + m_layers[i].GetBias();

			//Activation function
			for (int j = 0; j < current_output.rows(); ++j)
			{
				current_output(j) = sigmoid(current_output(j));
			}
			//std::cout << "current output: " << current_output << std::endl;
			prev_output = current_output;
		}

	}
	std::vector<float> *final_output = new std::vector<float>(prev_output.data(), prev_output.data() + prev_output.cols() * prev_output.rows());

	return *final_output;
}

NeuralNetwork::~NeuralNetwork()
{
	for (int i = 0; i < m_layers.size(); ++i)
	{
		delete m_layers[i].GetInputs();
	}
}

void NeuralNetwork::InitWeights()
{
	//srand((unsigned int)time(NULL));
	for (int i = 0; i < m_layers.size(); ++i)
	{
		if (i < m_layers.size() - 1)
		{
			int row_num = m_layers[i].GetInputs()->size();
			int col_num = m_layers[i + 1].GetInputs()->size();

			Eigen::MatrixXf weight_mat(col_num, row_num);
			weight_mat.setRandom();
			m_weights.emplace_back(weight_mat);
		}
	}
}

inline float NeuralNetwork::Relu(float num)
{
	return num < 0 ? 0 : num;
}

inline float NeuralNetwork::sigmoid(float num)
{
	return 1 / (1 + exp(-num));
}

inline float NeuralNetwork::dsigmoid(float num)
{
	return 0;
}


std::vector<float> NeuralNetwork::Feedforward()
{
	Eigen::MatrixXf prev_output;
	for (int i = 0; i < m_layers.size() - 1; ++i)
	{
		//construct current input vector and weight matrix by using the number of input as 
		//the row number and the number of next layer as the column number
		int row_num = m_layers[i].GetInputs()->size();
		int col_num = m_layers[i + 1].GetInputs()->size();

		Eigen::MatrixXf current_input(row_num, 1);
		Eigen::MatrixXf current_bias(m_layers[i + 1].GetInputs()->size(), 1);

		//srand((unsigned int)time(NULL));
		current_bias.setRandom();
		m_layers[i].SetBias(current_bias);

		Eigen::MatrixXf weight_mat(col_num, row_num);
		weight_mat.setOnes();

		Eigen::MatrixXf current_output(row_num, 1);

		if (i == 0)
		{
			float* input_array_raw = m_layers[i].GetInputs()->data();
			current_input = Eigen::Map<Eigen::MatrixXf>(input_array_raw, row_num, 1);
			//Input is the output of the 0th layer
			m_outputs.emplace_back(current_input);
		}
		else
		{
			current_input = prev_output;
		}
		current_output = m_weights[i] * current_input;

		//Add the bias
		current_output = current_output + m_layers[i].GetBias();
		//Activation function
		for (int j = 0; j < current_output.rows(); ++j)
		{
			current_output(j) = sigmoid(current_output(j));
		}
		m_outputs.emplace_back(current_output);
		//std::cout << "current output: " << current_output << std::endl;
		prev_output = current_output;

	}
	std::vector<float> final_output(prev_output.data(), prev_output.data() + prev_output.cols() * prev_output.rows());

	return final_output;
}

void NeuralNetwork::Train(std::vector<float> inputs, std::vector<float> targets)
{
	std::vector<float> output = Feedforward();

	Eigen::MatrixXf current_error;
	for (int i = m_outputs.size() - 1; i >= 0; --i)
	{
		//For the first time, error can be calculated directly from target - final output
		if (i == m_outputs.size() - 1)
		{
			//Calculate output error
			std::vector<float> error;
			for (int i = 0; i < targets.size(); ++i)
			{
				error.emplace_back(targets[i] - output[i]);
			}
			/*std::cout << "Output = " << output[0] << std::endl;
			std::cout << "Error = " << error[0] << std::endl;
			if (error[0] > 0) 
			{
				PRINT("ERROR > 0, Should INCREASE weights");
			}
			else 
			{
				PRINT("ERROR < 0, Should DECREASE weights");
			}*/

			float* error_raw = error.data();
			current_error = Eigen::Map<Eigen::MatrixXf>(error_raw, error.size(), 1);
		}
		//otherwise multiply current error with current weight to get the error of this layer
		else
		{
			current_error = m_weights[i].transpose() * current_error;
		}

		/*PRINT("Error:");
		PRINT(current_error);*/
		//Make a vector full of ones for below calculation of derivaitive
		Eigen::MatrixXf ones_vector(m_outputs[i].rows(), 1);
		ones_vector.setOnes();

		//Gradient of current output(derivaitive)
		Eigen::MatrixXf gradient =
			0.05 * current_error.cwiseProduct(m_outputs[i].cwiseProduct(ones_vector - m_outputs[i]));

		//The delta weight at this layer to propergate back
		Eigen::MatrixXf current_d_weight =
			 gradient * m_outputs[i - 1].transpose();

		/*PRINT("delta w:");
		PRINT(current_d_weight);*/

		//update current weight matrix
		m_weights[i-1] = m_weights[i-1] + current_d_weight;

		//Update bias
		Eigen::MatrixXf current_layer_bias = m_layers[i - 1].GetBias();
		m_layers[i-1].SetBias(current_layer_bias - gradient);

		if (current_d_weight(1) < 0)
			PRINT("weight decreased");
		//Prevent index out of bound error
		if (i - 1 == 0)
			break;
	}

	m_outputs.clear();
}

Layer::Layer(int shape) : m_shape(shape)
{
	m_layer_inputs = new std::vector<float>();
	for (int i = 0; i < m_shape; ++i)
	{
		m_layer_inputs->emplace_back(0.0f);
	}

	
	//m_layer_inputs->emplace_back(m_bias);
}
