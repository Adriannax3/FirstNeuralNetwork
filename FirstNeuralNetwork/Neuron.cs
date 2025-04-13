using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FirstNeuralNetwork
{
    public class Neuron
    {
        public double[] Weights;
        public double Bias;
        public double Output;
        public double Delta;

        public Neuron(int inputCount, Random random)
        {
            Weights = new double[inputCount];
            for(int i = 0; i < inputCount; i++)
            {
                Weights[i] = random.NextDouble() * 2 - 1;
            }
            Bias = random.NextDouble() * 2 - 1;
        }

        public double Activate(double[] inputs)
        {
            double sum = Bias;
            for(int i=0;i< inputs.Length; i++)
            {
                sum += inputs[i] * Weights[i];
            }
            Output = Sigmoid(sum);
            return Output;
        }

        private double Sigmoid(double x) {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}
