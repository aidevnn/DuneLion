using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Activations
{
    public class SigmoidActivation<U> : BaseActivation<U>
    {
        public SigmoidActivation()
        {
            funcExpr = ND.Sigmoid(xF);
            derivExpr = ND.DSigmoid(xDF);
            Name = "Sigmoid";
        }
    }
}
