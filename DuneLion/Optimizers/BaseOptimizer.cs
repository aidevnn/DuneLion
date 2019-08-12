using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Optimizers
{
    public abstract class BaseOptimizer<U>
    {
        protected BaseOptimizer()
        {
            weights = Variable.CreateNDarray<U>("w");
            grad = Variable.CreateNDarray<U>("g");
        }

        public string Name { get; protected set; }
        protected readonly NullaryExpr<U> weights, grad;
        protected Expr<U> optimizerExpr;

        public abstract BaseOptimizer<U> Clone();

        public abstract NDarray<U> Update(NDarray<U> w, NDarray<U> g);
    }
}
