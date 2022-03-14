// Wheel-NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <Eigen/Dense>
#include "NeuralNetwork.h"

int main()
{
    std::vector<Layer> layers;
    layers.emplace_back(3);
    layers.emplace_back(4);
    layers.emplace_back(4);
    
    std::vector<float> inputs = { 1,3,4 };

    NeuralNetwork nn(layers);
    nn.SetInput(inputs);
    nn.Feedforward();

}
