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
        public static double GradiantLogSigmoid(double x)
        {
            return x * (1 - x);
        }
        public static double GraduitHyperbolicTangtent(double x)
        {
         
            return 1 - x * x;
        }

        public static Func<double, double> GetGraduitForActivationFunction(Func<double, double> fn)
        {
            if (fn.Equals(((Func<double, double>)ActivationFunction.LogSigmoid))) return ActivationFunction.GradiantLogSigmoid;
            if (fn.Equals(((Func<double, double>)ActivationFunction.HyperbolicTangtent))) return ActivationFunction.GraduitHyperbolicTangtent;
            throw new NotImplementedException("can't found the gradian for the func " + fn.Method.Name);

        }



    }
}
