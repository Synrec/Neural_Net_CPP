#include <iostream>
#include "Headers/Neural_Network.h"

int main()
{
    std::vector<unsigned> topology = {2, 3, 1};
    NeuralNetwork Net(topology, 0.1);
    std::vector<std::vector<float>> target_inputs{
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f}
    };
    std::vector<std::vector<float>> target_outputs{
        {1.0f},
        {0.0f},
        {0.0f},
        {1.0f}
    };
    unsigned epoch = 100000;

    std::cout << "Training Started\n";
    for(unsigned i = 0; i < epoch; i++){
        unsigned index = rand() % 4;
        Net.feedForward(target_inputs[index]);
        Net.backPropagate(target_outputs[index]);
    }
    std::cout << "Training Completed\n";
    for(auto input : target_inputs){
        Net.feedForward(input);
        auto preds = Net.getPrediction();
        std::cout << input[0] << " ," << input[1] << " -> " << preds[0] << std::endl;
    }
    return 0;
}