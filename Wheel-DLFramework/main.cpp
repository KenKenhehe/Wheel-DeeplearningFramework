// Wheel-NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <Eigen/Dense>
#include "NeuralNetwork.h"

int main()
{
    std::vector<Layer> layers;
    layers.emplace_back(2);
    layers.emplace_back(2);
    layers.emplace_back(2);
    
    std::vector<float> inputs = { 1,1};
    std::vector<float> target = { 1,0 };

    NeuralNetwork nn(layers);
    nn.SetInput(inputs);
    //nn.Feedforward();
    nn.Train(inputs, target);
}
