using System;
namespace NeuroEvolutionEngine.Core.Math
{
    public static class ActivationFunction
    {
        // transform value from [0,1] to [-1,1]
        // not really an activation function!!
        public static double LinearCalibration(double i)
        {
            return i * 2 - 1;
        }

        public static double LogSigmoid(double x)
        {
            if (x < -45.0) return 0.0;
            else if (x > 45.0) return 1.0;
            else return 1.0 / (1.0 + System.Math.Exp(-x));
        }

        public static double HyperbolicTangtent(double x)
        {
            if (x < -45.0) return -1.0;
            else if (x > 45.0) return 1.0;
            else return System.Math.Tanh(x);
        }
        public static double GradientLogSigmoid(double x)
        {
            return x * (1 - x);
        }
        public static double GradientHyperbolicTangtent(double x)
        {
         
            return 1 - x * x;
        }


        public static double Relu(double x)
        {
            return System.Math.Max(0, x);
        }
        public static double GradientRelu(double x)
        {
            return x;
        }

        public static Func<double, double> GetGraduitForActivationFunction(Func<double, double> fn)
        {
            if (fn.Equals(((Func<double, double>)ActivationFunction.LogSigmoid))) return ActivationFunction.GradientLogSigmoid;
            if (fn.Equals(((Func<double, double>)ActivationFunction.HyperbolicTangtent))) return ActivationFunction.GradientHyperbolicTangtent;
            if (fn.Equals(((Func<double, double>)ActivationFunction.Relu))) return ActivationFunction.GradientRelu;
            throw new NotImplementedException("can't found the gradian for the func " + fn.Method.Name);

        }

        public static Func<double, double> GetActivationFunction(string fn)
        {
            if (fn.Equals(((Func<double, double>)ActivationFunction.LogSigmoid).Method.Name)) return ActivationFunction.LogSigmoid;
            if (fn.Equals(((Func<double, double>)ActivationFunction.HyperbolicTangtent).Method.Name)) return ActivationFunction.HyperbolicTangtent;
            if (fn.Equals(((Func<double, double>)ActivationFunction.Relu).Method.Name)) return ActivationFunction.Relu;
            throw new NotImplementedException("can't found the the func " + fn);

        }

    }
}
