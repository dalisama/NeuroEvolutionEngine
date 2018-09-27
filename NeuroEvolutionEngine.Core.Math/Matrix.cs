using csMatrix;
using System;

namespace NeuroEvolutionEngine.Core.Math
{
    public static class MatrixExtension
    {

        public static void ApplyFn(this Matrix mtx, Func<double, double> fn)
        {
            for (int i = 0; i < mtx.Columns; i++)
            {
                for (int j = 0; j < mtx.Rows; j++)
                {
                    mtx[j, i] = fn(mtx[j, i]);
                }
            }
        }

        public static void  MultiplyElementByElement(this Matrix mtx1, Matrix mtx2)
        {
            if (mtx1.Columns == mtx2.Columns && mtx2.Rows == mtx1.Rows)
            {
           
                for (int i = 0; i < mtx1.Columns; i++)
                {
                    for (int j = 0; j < mtx1.Rows; j++)
                    {
                        mtx1[j, i] = mtx1[j, i] * mtx2[j, i];
                    }
                }
               
            }
            else
            {
                throw new Exception("Matrix must have the same dimentions");
            }
        }

        ///todo import export matrix
        ///todo display matrix


    }
}
