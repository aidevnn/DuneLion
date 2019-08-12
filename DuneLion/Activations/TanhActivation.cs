using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Activations
{
    public class TanhActivation<U> : BaseActivation<U>
    {
        public TanhActivation()
        {
            funcExpr = ND.Tanh(xF);
            derivExpr = ND.DTanh(xDF);
            Name = "Tanh";
        }
    }
}
