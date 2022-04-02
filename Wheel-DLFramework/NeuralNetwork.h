#pragma once
#ifndef NN_H
#define NN_H
#define PRINT(x) std::cout << x << std::endl

#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <ctime>

class Layer 
{
public:
	Layer(int shape);

	Eigen::MatrixXf GetBias() { return m_bias; }
	void SetBias(Eigen::MatrixXf bias) { m_bias = bias; }
	std::vector<float>* GetInputs() { return m_layer_inputs; }
private:
	Eigen::MatrixXf m_bias;
	int m_shape;
	std::vector<float>* m_layer_inputs;
	//TODO:function pointer member for activation function
};

class NeuralNetwork
{
public:
	NeuralNetwork(std::vector<Layer> layers);
	~NeuralNetwork();

	std::vector<float> Feedforward();
	void Train(std::vector<float> inputs, std::vector<float> targets);
	void SetInput(std::vector<float> inputs);
	std::vector<float> predict(std::vector<float> inputs);
private:
	std::vector<Layer> m_layers;
	std::vector<Eigen::MatrixXf> m_weights;
	//Computed output for each layer
	std::vector<Eigen::MatrixXf> m_outputs;
private:
	void InitWeights();

	inline float Relu(float num);
	inline float sigmoid(float num);
	inline float dsigmoid(float num);
};

#endif // !NN_H



