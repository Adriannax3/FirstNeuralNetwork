using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FirstNeuralNetwork;

namespace FirstNeuralNetwork
{
    public class Program
    {
        public static void Main() {
            var nn = new NeuralNetwork(2, 2, 1);

            double[][] inputs =
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 },
            };

            double[][] targets =
            {
                new double [] { 0 },
                new double [] { 1 },
                new double [] { 1 },
                new double [] { 1 }
            };

            for (int epoch = 0; epoch < 10000; epoch++)
            {
                for (int i = 0; i < inputs.Length; i++)
                    nn.Train(inputs[i], targets[i]);
            }
            Console.WriteLine("Results:");
            foreach (var input in inputs)
            {
                var wynik = nn.Predict(input)[0];
                Console.WriteLine($"{input[0]} OR {input[1]} = {Math.Round(wynik)} ({wynik:F4})");
            }
        }
    }
}
