using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Activations
{
    public abstract class BaseActivation<U>
    {
        protected BaseActivation()
        {
            xF = Variable.CreateNDarray<U>("x_f");
            xDF = Variable.CreateNDarray<U>("x_df");
        }

        protected Expr<U> funcExpr, derivExpr;
        protected readonly Expr<U> xF, xDF;

        public string Name { get; protected set; }
        public NDarray<U> Func(NDarray<U> X) => funcExpr.Evaluate(("x_f", X));
        public NDarray<U> Derivative(NDarray<U> X) => derivExpr.Evaluate(("x_df", X));
    }
}
