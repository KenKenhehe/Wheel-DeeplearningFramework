#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers) : m_layers(layers)
{
	InitWeights();
}

void NeuralNetwork::SetInput(std::vector<float> inputs)
{
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
				//Input is the output of the 0th layer
			}
			else
			{
				current_input = prev_output;
			}
			current_output = m_weights[i] * current_input;
			//Activation function
			for (int i = 0; i < current_output.rows(); ++i)
			{
				current_output(i) = Relu(current_output(i));
			}
			//std::cout << "current output: " << current_output << std::endl;
			prev_output = current_output;
		}

	}
	std::vector<float> final_output(prev_output.data(), prev_output.data() + prev_output.cols() * prev_output.rows());

	return final_output;
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
	for (int i = 0; i < m_layers.size(); ++i)
	{
		if (i < m_layers.size() - 1)
		{
			int row_num = m_layers[i].GetInputs()->size();
			int col_num = m_layers[i + 1].GetInputs()->size();

			Eigen::MatrixXf weight_mat(col_num, row_num);
			weight_mat.setOnes();
			m_weights.emplace_back(weight_mat);
		}
	}
}

inline float NeuralNetwork::Relu(float num)
{
	return num < 0 ? 0 : num;
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
		//Activation function
		for (int i = 0; i < current_output.rows(); ++i)
		{
			current_output(i) = Relu(current_output(i));
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

			float* error_raw = error.data();
			current_error = Eigen::Map<Eigen::MatrixXf>(error_raw, error.size(), 1);
		}
		//otherwise multiply current error with current weight to get the error of this layer
		else
		{
			current_error = m_weights[i - 1].transpose() * current_error;
		}

		//Make a vector full of ones for below calculation of derivaitive
		Eigen::MatrixXf ones_vector(m_outputs[i].rows(), 1);
		ones_vector.setOnes();

		//Gradient of current output(derivaitive)
		Eigen::MatrixXf gradient =
			current_error.cwiseProduct(m_outputs[i].cwiseProduct(ones_vector - m_outputs[i]));

		//The delta weight at this layer to propergate back
		Eigen::MatrixXf current_d_weight =
			0.02 * gradient * m_outputs[i - 1].transpose();

		PRINT(current_d_weight);

		//update current weight matrix
		m_weights[i - 1] = m_weights[i - 1] - current_d_weight;

		//Prevent index out of bound error
		if (i - 1 == 0)
			break;
	}
}

Layer::Layer(int shape, int bias) : m_shape(shape), m_bias(bias)
{
	m_layer_inputs = new std::vector<float>();
	for (int i = 0; i < m_shape; ++i)
	{
		m_layer_inputs->emplace_back(0.0f);
	}
}
