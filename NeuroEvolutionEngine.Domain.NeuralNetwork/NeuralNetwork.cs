using csMatrix;
using NeuroEvolutionEngine.Core.Math;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuroEvolutionEngine.Domain.NeuralNetwork
{
    public class NeuralNetwork
    {

        public int InputLayer { get; set; }
        public int[] HiddenLayer { get; set; }
        public int OutputLayer { get; set; }
        public Matrix[] Weights { get; set; }
        public Matrix[] Bias { get; set; }
        public Func<double, double> ActivationFuntion { get; set; }



        public NeuralNetwork(int inputLayer, int[] hiddenLayer, int outputLayer, Func<double, double> activationFuntion)
        {
            ActivationFuntion = activationFuntion;
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

        public NeuralNetwork(NeuralNetwork nn)
        {
            Bias = nn.Bias.Select(x => new Matrix(x)).ToArray();
            Weights = nn.Weights.Select(x => new Matrix(x)).ToArray();
            HiddenLayer = nn.HiddenLayer;
            OutputLayer = nn.OutputLayer;
            InputLayer = nn.InputLayer;
            ActivationFuntion = nn.ActivationFuntion;
        }

        public NeuralNetwork(Matrix[] weights, Matrix[] bias, int[] hiddenLayer, Func<double, double> activationFunction, int intputLayer, int outputLayer)
        {
            Weights = weights;
            Bias = bias;
            HiddenLayer = hiddenLayer;
            this.ActivationFuntion = activationFunction;
            this.InputLayer = intputLayer;
            OutputLayer = outputLayer;
        }

        public Matrix FeedForward(Matrix input, out Matrix[] tmpOutputs)
        {
            tmpOutputs = new Matrix[Weights.Length];
            for (int i = 0; i < Weights.Length; i++)
            {
                input = Weights[i] * input + Bias[i];
                input.ApplyFn(ActivationFuntion);
                tmpOutputs[i] = input;
            }

            return input;
        }

        public Matrix Prediction(Matrix input)
        {

            for (int i = 0; i < Weights.Length; i++)
            {
                input = Weights[i] * input + Bias[i];
                input.ApplyFn(ActivationFuntion);

            }

            return input;
        }

        public void Export(string path, string fileName)

        {
            var fullPath = Path.Combine(path, fileName);
            if (File.Exists(fullPath)) File.Delete(Path.Combine(path, fullPath));

            var jsonObject = new List<string>();
            JsonSerializer serializer = new JsonSerializer();
            for (int i = 0; i < Weights.Length; i++)
            {
                jsonObject.Add(JsonConvert.SerializeObject(this.Weights[i].Data));
                jsonObject.Add(JsonConvert.SerializeObject(this.Weights[i].Rows));
                jsonObject.Add(JsonConvert.SerializeObject(this.Weights[i].Columns));

            }
            jsonObject.Add("#");
            for (int i = 0; i < Bias.Length; i++)
            {
                jsonObject.Add(JsonConvert.SerializeObject(this.Bias[i].Data));
                jsonObject.Add(JsonConvert.SerializeObject(this.Bias[i].Rows));
                jsonObject.Add(JsonConvert.SerializeObject(this.Bias[i].Columns));

            }
            jsonObject.Add("#");
            jsonObject.Add(JsonConvert.SerializeObject(this.HiddenLayer));
            jsonObject.Add((ActivationFuntion.Method.Name));
            jsonObject.Add(InputLayer.ToString());
            jsonObject.Add(OutputLayer.ToString());
            File.WriteAllLines(fullPath, jsonObject);


        }
        public static NeuralNetwork Import(string path, string fileName)
        {

            var fullPath = Path.Combine(path, fileName);
            var jsonObject = File.ReadAllLines(fullPath);
            var index = 0;
            var tmpWeights = new List<Matrix>();
            while (jsonObject[index] != "#")
            {
                var data= JsonConvert.DeserializeObject<double[]>(jsonObject[index++]);
                var row = JsonConvert.DeserializeObject<int>(jsonObject[index++]);
                var column = JsonConvert.DeserializeObject<int>(jsonObject[index++]);
                tmpWeights.Add(new Matrix(row, column, data));
            }
            var weights = tmpWeights.ToArray();
            index++;

            var tmpBias = new List<Matrix>();
            while (jsonObject[index] != "#")
            {
                var data = JsonConvert.DeserializeObject<double[]>(jsonObject[index++]);       
                var row = JsonConvert.DeserializeObject<int>(jsonObject[index++]);
                var column = JsonConvert.DeserializeObject<int>(jsonObject[index++]);
                tmpBias.Add(new Matrix(row, column, data));
            }
            var bias = tmpBias.ToArray();
            index++;

            var hiddenLayer = JsonConvert.DeserializeObject<int[]>(jsonObject[index++]);
            var activationFunction = ActivationFunction.GetActivationFunction (jsonObject[index++]);
            var intputLayer = JsonConvert.DeserializeObject<int>(jsonObject[index++]);
            var outputLayer = JsonConvert.DeserializeObject<int>(jsonObject[index]);
            return new NeuralNetwork(weights, bias, hiddenLayer, activationFunction, intputLayer, outputLayer);
        }



    }
}
