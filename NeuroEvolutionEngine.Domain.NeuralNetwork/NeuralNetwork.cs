using csMatrix;
using NeuroEvolutionEngine.Core.Math;
using System;

namespace NeuroEvolutionEngine.Domain.NeuralNetwork
{
    public class NeuralNetwork
    {
        public int InputLayer { get; set; }
        public int[] HiddenLayer { get; set; }
        public int OutputLayer { get; set; }
        public Matrix[] Weights { get; set; }
        public Matrix[] Bias { get; set; }
        public double LearningRate { get; set; }


        public NeuralNetwork(int inputLayer, int[] hiddenLayer, int outputLayer)
        {
            InputLayer = inputLayer;
            HiddenLayer = hiddenLayer;
            OutputLayer = outputLayer;

            //initilize the weight
            Weights = new Matrix[HiddenLayer.Length + 1];
            Weights[0] = new Matrix(HiddenLayer[0], InputLayer);
            Weights[0].Rand();
            Weights[0].ApplyFn(ActivationFunction.LinearCalibration);

            for (int i = 1; i < Weights.Length - 1; i++)
            {
                Weights[i] = new Matrix(HiddenLayer[i], HiddenLayer[i - 1]);
                Weights[i] = Weights[i].Rand();
                Weights[i].ApplyFn(ActivationFunction.LinearCalibration);

            }
            Weights[Weights.Length - 1] = new Matrix(OutputLayer, HiddenLayer[HiddenLayer.Length - 1]);
            Weights[Weights.Length - 1].Rand();
            Weights[Weights.Length - 1].ApplyFn(ActivationFunction.LinearCalibration);
            //initilize the bias
            Bias = new Matrix[hiddenLayer.Length + 1];
            for (int i = 0; i < Bias.Length - 1; i++)
            {
                Bias[i] = new Matrix(hiddenLayer[i], 1);
                Bias[i] = Bias[i].Rand();
                Bias[i].ApplyFn(ActivationFunction.LinearCalibration);
            }

            Bias[Bias.Length - 1] = new Matrix(outputLayer, 1);
            Bias[Bias.Length - 1] = Bias[Bias.Length - 1].Rand();
            Bias[Bias.Length - 1].ApplyFn(ActivationFunction.LinearCalibration);

        }





        public Matrix FeedForward(Matrix input, Func<double, double> fn, out Matrix[] tmpOutputs)
        {
            tmpOutputs = new Matrix[Weights.Length];
            for (int i = 0; i < Weights.Length; i++)
            {
                input = Weights[i] * input + Bias[i];
                input.ApplyFn(fn);
                tmpOutputs[i] = input;
            }

            return input;
        }

        public Matrix FeedForward(Matrix input, Func<double, double> fn)
        {

            for (int i = 0; i < Weights.Length; i++)
            {
                input = Weights[i] * input + Bias[i];
                input.ApplyFn(fn);

            }

            return input;
        }



    }
}
