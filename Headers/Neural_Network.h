#include "Matrix.h"
#include <cstdlib>

inline float Sigmoid(float x)
{
    return 1.0 / (1 + exp(-x));
}

inline float DSigmoid(float x){
    return x * (1 - x);
}

class NeuralNetwork
{
    public:
    std::vector<unsigned> _topology;
    std::vector<Matrix> _weightMatrices;
    std::vector<Matrix> _valueMatrices;
    std::vector<Matrix> _biasMatrices;
    float _learning_rate;
        NeuralNetwork(std::vector<unsigned> topology, float learning_rate = 0.1f)
            :_topology(topology), _weightMatrices({}), _valueMatrices({}), _biasMatrices({}), _learning_rate(learning_rate)
        {
            for(unsigned i = 0; i < topology.size() - 1; i++){
                Matrix weightMatrix(topology[i + 1], topology[i]);
                weightMatrix = weightMatrix.applyFunction(
                    [](const float &f){
                        return (float)rand() / RAND_MAX;
                    } 
                );
                _weightMatrices.push_back(weightMatrix);
                Matrix biasMatrix(topology[i + 1], 1);
                biasMatrix = biasMatrix.applyFunction(
                    [](const float &f){
                        return (float)rand() / RAND_MAX;
                    } 
                );
                _biasMatrices.push_back(biasMatrix);
            }
            _valueMatrices.resize(topology.size());
        }

        bool feedForward(std::vector<float> input)
        {
            if(input.size() != _topology[0]){
                return false;
            }
            Matrix values(input.size(), 1);
            values._vals = input;
            for(unsigned i = 0; i < _weightMatrices.size(); i++){
                _valueMatrices[i] = values;
                values = values.multiply(_weightMatrices[i]);
                values = values.add(_biasMatrices[i]);
                values = values.applyFunction(Sigmoid);
            }
            _valueMatrices[_weightMatrices.size()] = values;
            return true;
        }

        bool backPropagate(std::vector<float> targetOutput)
        {
            if(targetOutput.size() != _topology.back()){
                return false;
            }
            Matrix errors(targetOutput.size(), 1);
            errors._vals = targetOutput;
            errors = errors.add(_valueMatrices.back().negative());
            for(int i = _weightMatrices.size() - 1; i >= 0; i--){
                Matrix prevErrors = errors.multiply(_weightMatrices[i].transpose());
                Matrix dOutputs = _valueMatrices[i + 1].applyFunction(DSigmoid);
                Matrix gradients = errors.multiply_elements(dOutputs);
                gradients = gradients.multiply_scalar(_learning_rate);
                Matrix weightGradients = _valueMatrices[i].transpose().multiply(gradients);
                _weightMatrices[i] = _weightMatrices[i].add(weightGradients);
                _biasMatrices[i] = _biasMatrices[i].add(gradients);
                errors = prevErrors;
            }
            return true;
        }

        std::vector<float> getPrediction()
        {
            return _valueMatrices.back()._vals;
        }
};