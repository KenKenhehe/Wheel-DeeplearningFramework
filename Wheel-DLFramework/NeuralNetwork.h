#pragma once
#ifndef NN_H
#define NN_H

#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Dense>

class Layer 
{
public:
	Layer(int shape, int bias = 1);

	float GetBias() { return m_bias; }
	std::vector<float>* GetInputs() { return m_layer_inputs; }
private:
	float m_bias;
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
	void SetInput(std::vector<float> inputs);
private:
	std::vector<Layer> m_layers;
	std::vector<Eigen::MatrixXf> m_weights;
private:
	void InitWeights();
};

#endif // !NN_H



