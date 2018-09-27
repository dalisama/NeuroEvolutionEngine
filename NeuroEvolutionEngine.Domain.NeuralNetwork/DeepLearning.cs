﻿using csMatrix;
using NeuroEvolutionEngine.Core.Math;
using System;

namespace NeuroEvolutionEngine.Domain.NeuralNetwork
{
    public class DeepLearning
    {


        public NeuralNetwork Brain { get; set; }
        public Func<double, double> ActivationFuntion { get; set; }
        public double LearningRate { get; set; }


        private Matrix[] _tmpOutputs;


        public DeepLearning(NeuralNetwork brain, Func<double, double> activationFuntion, double learningRate)
        {
            Brain = brain;
            ActivationFuntion = activationFuntion;
            LearningRate = learningRate;
        }

        public Matrix[] CalculateMatrixError(Matrix inputs, Matrix target)
        {
            var output = Brain.FeedForward(inputs, ActivationFuntion, out _tmpOutputs);
            var ErrorsMatrix = new Matrix[Brain.HiddenLayer.Length + 1];

            // calculate error at output node of the NN
            ErrorsMatrix[Brain.HiddenLayer.Length] = target - output;
            for (int i = Brain.HiddenLayer.Length - 1; i >= 0; i--)
            {
                ErrorsMatrix[i] = Matrix.Transpose(Brain.Weights[i + 1]) * ErrorsMatrix[i + 1];
            }

            return ErrorsMatrix;
        }
        public void Train(Matrix inputs, Matrix target)
        {
            LearningRate = 1;
            var errorMatrix = CalculateMatrixError(inputs, target);
            for (int i = 0; i < Brain.Weights.Length; i++)
            {
                // calculate gradient
                var deltaWeight = new Matrix(_tmpOutputs[i]);
                deltaWeight.ApplyFn(ActivationFunction.GetGraduitForActivationFunction(ActivationFuntion));
                deltaWeight.MultiplyElementByElement(errorMatrix[i]);
                deltaWeight = deltaWeight * LearningRate;
                // adjust biasis delta bias is the gradiant
                Brain.Bias[i] = Brain.Bias[i] + deltaWeight;
                // calculate delta weight
                if (i == 0)
                {
                    deltaWeight = deltaWeight * Matrix.Transpose(inputs);
                }
                else
                {
                    deltaWeight = deltaWeight * Matrix.Transpose(_tmpOutputs[i - 1]);
                }
                // update weight
                Brain.Weights[i] = Brain.Weights[i] + deltaWeight;

            }
        }



    }
}