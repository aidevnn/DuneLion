using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Accuracies
{
    public abstract class BaseAccuracy<U>
    {
        protected BaseAccuracy()
        {
            y0 = Variable.CreateNDarray<U>("y");
            p0 = Variable.CreateNDarray<U>("p");
        }

        public string Name { get; protected set; }
        protected readonly NullaryExpr<U> y0, p0;
        protected Expr<double> accExpr;

        public abstract NDarray<double> Acc(NDarray<U> y, NDarray<U> p);
    }
}
