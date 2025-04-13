using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FirstNeuralNetwork
{
    public class NeuralNetwork
    {
        private Neuron[] hidden;
        private Neuron[] output;
        private double learningRate = 0.5;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            var rnd = new Random();
            hidden = new Neuron[inputSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                hidden[i] = new Neuron(inputSize, rnd);
            }

            output = new Neuron[outputSize];
            for(int i = 0;i < outputSize; i++)
            {
                output[i] = new Neuron(hiddenSize, rnd);
            }
        }

        public double[] Predict(double[] inputs)
        {
            var hiddenOutputs = hidden.Select(n => n.Activate(inputs)).ToArray();
            return output.Select(n => n.Activate(hiddenOutputs)).ToArray();
        }

        public void Train(double[] inputs, double[] targets)
        {
            var hiddenOutputs = hidden.Select(n => n.Activate(inputs)).ToArray();
            var finalOutputs = output.Select(n=> n.Activate(hiddenOutputs)).ToArray();

            for (int i = 0; i < output.Length; i++) { 
                double error = targets[i] - finalOutputs[i];
                output[i].Delta = error * finalOutputs[i] * (1-finalOutputs[i]);
            }

            for (int i = 0; i < hidden.Length; i++) { 
                double error = output.Sum(o => o.Weights[i] * o.Delta);
                hidden[i].Delta = error * hidden[i].Output * (1 - hidden[i].Output);
            }

            for (int i = 0; i < output.Length; i++) {
                for (int j = 0; j < inputs.Length; j++)
                {
                    output[i].Weights[j] += learningRate * output[i].Delta * hidden[j].Output;
                }
                output[i].Bias += learningRate * output[i].Delta;
            }

            for (int i = 0; i < hidden.Length; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    hidden[i].Weights[j] += learningRate * hidden[i].Delta * inputs[j];
                }
                hidden[i].Bias += learningRate * hidden[i].Delta;
            }
        }
    }
}
