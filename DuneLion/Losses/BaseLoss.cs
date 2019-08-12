using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Losses
{
    public abstract class BaseLoss<U>
    {
        protected BaseLoss()
        {
            y0 = Variable.CreateNDarray<U>("y");
            p0 = Variable.CreateNDarray<U>("p");
        }

        public string Name { get; protected set; }
        protected readonly NullaryExpr<U> y0, p0;
        protected Expr<double> lossExpr;
        protected Expr<U> gradExpr;

        public abstract NDarray<double> Loss(NDarray<U> y, NDarray<U> p);
        public abstract NDarray<U> Grad(NDarray<U> y, NDarray<U> p);
    }
}
