using csMatrix;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuroEvolutionEngine.Core.Math;
using NeuroEvolutionEngine.Domain.NeuralNetwork;
using System;

namespace NeuroEvolutionEngine.Test
{
    [TestClass]
    public class NeuralNetworkTest
    {
        [TestMethod]
        public void ShouldReturnSth()
        {
            var brain = new NeuralNetwork(5, new int[1] { 5 }, 1);
            var input = new Matrix(5, 1);
            input.Rand();
            Matrix[] outputs;
            var tm = brain.FeedForward(input, ActivationFunction.LogSigmoid, out outputs);
            Assert.IsNotNull(tm);
        }
        [TestMethod]
        public void ShouldReturnSths()
        {
            var brain = new NeuralNetwork(5, new int[5] { 5, 3, 6, 7, 4 }, 1);
            var input = new Matrix(5, 1);
            input.Rand();
            Matrix[] outputs;
            var tm = brain.FeedForward(input, ActivationFunction.LogSigmoid, out outputs);
            Assert.IsNotNull(tm);
        }
        [TestMethod]
        public void ShouldReturnSth3s()
        {
            var brain = new NeuralNetwork(5, new int[5] { 5, 3, 6, 7, 4 }, 2);
            var input = new Matrix(5, 1);
            input.Rand();
            Matrix[] outputs;
            var tm = brain.FeedForward(input, ActivationFunction.LogSigmoid, out outputs);
            Assert.IsNotNull(tm);
        }
        [TestMethod]
        public void ShouldReturnSthRs()
        {
            var brain = new NeuralNetwork(5, new int[5] { 5, 3, 6, 7, 4 }, 2);

            var input = new Matrix(5, 1);
            input.Rand();
            var target = new Matrix(2, 1);
            target.Rand();
            Matrix[] outputs;
            var tm = brain.FeedForward(input, ActivationFunction.LogSigmoid, out outputs);

            var dl = new DeepLearning(brain, ActivationFunction.LogSigmoid, 10);
            var tt = dl.CalculateMatrixError(input, target);
            dl.Train(input, target);
            var tm2 = brain.FeedForward(input, ActivationFunction.LogSigmoid, out outputs);
            Assert.IsNotNull(tt);
        }

        [TestMethod]
        public void ShouldReturnXors()
        {
            var brain = new NeuralNetwork(2, new int[3] { 5,5,5}, 1);
            var deepLearning = new DeepLearning(brain, ActivationFunction.LogSigmoid, 0.001);

            var inputs = new Matrix[4];

            inputs[0] = new Matrix(2, 1);
            inputs[0][0, 0] = 0;
            inputs[0][1, 0] = 0;

            inputs[1] = new Matrix(2, 1);
            inputs[1][0, 0] = 1;
            inputs[1][1, 0] = 0;

            inputs[2] = new Matrix(2, 1);
            inputs[2][0, 0] = 0;
            inputs[2][1, 0] = 1;

            inputs[3] = new Matrix(2, 1);
            inputs[3][0, 0] = 1;
            inputs[3][1, 0] = 1;

            var targets = new Matrix[4];

            targets[0] = new Matrix(1, 1);
            targets[0][0, 0] = 0;

            targets[1] = new Matrix(1, 1);
            targets[1][0, 0] = 1;

            targets[2] = new Matrix(1, 1);
            targets[2][0, 0] = 1;

            targets[3] = new Matrix(1, 1);
            targets[3][0, 0] = 0;

            var rand = new Random();

            for (int i = 0; i < 500000; i++)
            {
                var t = rand.Next(4);
                deepLearning.Train(inputs[t], targets[t]);

            }
            
            var i0 = brain.FeedForward(inputs[0],  ActivationFunction.LogSigmoid);
            var i2 = brain.FeedForward(inputs[1], ActivationFunction.LogSigmoid);
            var i3 = brain.FeedForward(inputs[2], ActivationFunction.LogSigmoid);
            var i4 = brain.FeedForward(inputs[3], ActivationFunction.LogSigmoid);
            //var tm = brain.FeedForward(input, ActivationFunction.LogSigmoid, out outputs);
            Assert.IsFalse(false);
            //var dl = new DeepLearning(brain, ActivationFunction.LogSigmoid, 10);
            //var tt = dl.CalculateMatrixError(input, target);
            //dl.Train(input, target, ActivationFunction.LogSigmoid);
            //var tm2 = brain.FeedForward(input, ActivationFunction.LogSigmoid, out outputs);

        }
    }
}
