#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers): m_layers(layers)
{
	
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

std::vector<float> NeuralNetwork::Feedforward()
{
	InitWeights();

	
	Eigen::MatrixXf prev_output;
	for (int i = 0; i < m_layers.size(); ++i)
	{
		if (i < m_layers.size() - 1)
		{
			//Eigen::Matrix<float, m_layers[i].NodeSize()>
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
			std::cout << "current output: " << current_output << std::endl;
			prev_output = current_output;
			std::cout << "temp output: " << prev_output << std::endl;
		}

	}
	std::vector<float> final_output(prev_output.data(), prev_output.data() + prev_output.cols() * prev_output.rows());

	return final_output;
}

Layer::Layer(int shape, int bias): m_shape(shape), m_bias(bias)
{
	m_layer_inputs = new std::vector<float>();
	for (int i = 0; i < m_shape; ++i) 
	{
		m_layer_inputs->emplace_back(0.0f);
	}
}
