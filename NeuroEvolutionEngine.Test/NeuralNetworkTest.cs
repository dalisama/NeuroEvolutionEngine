using csMatrix;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuroEvolutionEngine.Core.Math;
using NeuroEvolutionEngine.Domain.NeuralNetwork;
using System;
using System.IO;
using System.Linq;

namespace NeuroEvolutionEngine.Test
{
    [TestClass]
    public class NeuralNetworkTest
    {


        [TestMethod]
        public void ShouldTrainForXor()
        {


            var brain = new NeuralNetwork(2, new int[3] { 5, 5, 5 }, 1, ActivationFunction.LogSigmoid);
            var deepLearning = new DeepLearning(brain, 0.00001);

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

            for (int i = 0; i < 10000; i++)
            {
                var t = rand.Next(4);
                deepLearning.Train(inputs[t], targets[t]);

            }

            var i0 = brain.Prediction(inputs[0]);// epected 0
            var i2 = brain.Prediction(inputs[1]);// expected 1
            var i3 = brain.Prediction(inputs[2]);//  expected 1
            var i4 = brain.Prediction(inputs[3]);//expected 0

            Assert.IsFalse(false);


        }

        private void GetDataFromMnsit(string path, out Matrix[] trainingSet, out Matrix[] TargetSet)
        {
            string[] lines = File.ReadAllLines(path);
            trainingSet = new Matrix[lines.Length];
            TargetSet = new Matrix[lines.Length];


            for (int i = 0; i < lines.Length; i++)
            {
                var tmp = Array.ConvertAll(lines[i].Split(','), s => double.Parse(s));
                TargetSet[i] = new Matrix(10, 1);
                TargetSet[i].Zeros();

                TargetSet[i][Convert.ToInt32(tmp[0]), 0] = 1;

                tmp = tmp.Skip(1).ToArray();
                trainingSet[i] = new Matrix(tmp.Length, 1, tmp);
                trainingSet[i] = trainingSet[i] / 255;




            }
        }
        [TestMethod]
        public void ShouldExportImportSameBrain()
        {
            Matrix[] testSet;
            Matrix[] testTraining;
            GetDataFromMnsit(@".\Mnist dataSet\mnistTest.csv", out testSet, out testTraining);

            var brain = new NeuralNetwork(testSet[0].Rows, new int[1] { 60 }, 10, ActivationFunction.LogSigmoid);
            brain.Export(@"c:\", "testttt");
            var deepLearning = new DeepLearning(brain, 0.001);

            var acc1 = calculateAccurcy(testSet, testTraining, brain);
            var tmp = NeuralNetwork.Import(@"c:\", "testttt");
            var acc2 = calculateAccurcy(testSet, testTraining, tmp);

            Assert.AreEqual(acc1, acc2);

        }

        [TestMethod]
        public void ShouldTrainForMnist()
        {

            // training data



            Matrix[] trainingSet;
            Matrix[] targetTraining;
            GetDataFromMnsit(@".\Mnist dataSet\mnistyTrain.csv", out trainingSet, out targetTraining);


            Matrix[] testSet;
            Matrix[] testTraining;
            GetDataFromMnsit(@".\Mnist dataSet\mnistTest.csv", out testSet, out testTraining);

            var brain = new NeuralNetwork(trainingSet[0].Rows, new int[1] { 60 }, 10, ActivationFunction.LogSigmoid);
           // brain = NeuralNetwork.Import(@"c:\", "Mnisty.json");
            var deepLearning = new DeepLearning(brain, 0.001);


            var MaxAccurcy = calculateAccurcy(testSet, testTraining, brain);

       
            System.Diagnostics.Debug.WriteLine("initial accurcy: " + MaxAccurcy);
            for (int j = 0; j < 10000; j++)
            {

                deepLearning.BulkTraining(trainingSet, targetTraining);


                var Currentaccurcy = calculateAccurcy(testSet, testTraining, brain);

                if (Currentaccurcy > MaxAccurcy)
                {
                    MaxAccurcy = Currentaccurcy;
                    brain.Export(@"c:\", "Mnisty.json");
                }


                System.Diagnostics.Debug.WriteLine("cycle " + j + " current " + Currentaccurcy + " max accurcy " + MaxAccurcy);

            }

            Assert.IsFalse(false);


        }

        private double calculateAccurcy(Matrix[] mSet, Matrix[] mLabel, NeuralNetwork brain)
        {

            var count = 0;
            var t1 = new Matrix(10, 1);
            var t11 = new Matrix(10, 1);
            t1.Zeros();
            for (int i = 0; i < mSet.Length; i++)
            {
                var guessMatrix = brain.Prediction(mSet[i]);
                var tmp = guessMatrix.Data.Max();
                var guess = guessMatrix.Data.ToList().IndexOf(tmp);

                tmp = mLabel[i].Data.Max();
                var result = mLabel[i].Data.ToList().IndexOf(tmp);
                if (guess == result)
                {
                    t1[guess, 0] = t1[guess, 0] + 1;
                    count++;
                }
                else
                {
                    t11[guess, 0] = t11[guess, 0] + 1;
                }

            }





            return ((double)count * 100) / mSet.Length;
        }

    }
}
