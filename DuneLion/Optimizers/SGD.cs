using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Optimizers
{
    public class SGD<U> : BaseOptimizer<U>
    {
        public SGD(double lr = 0.01, double momentum = 0.0)
        {
            this.lr = lr;
            this.momentum = momentum;
            Name = $"SGD lr[{lr}]" + (Math.Abs(momentum) < 1e-6 ? "" : $" mom[{momentum}]");

            wUpt = Variable.CreateNDarray<U>("w_upt");
            wUptExpr = wUpt * momentum + grad * (1 - momentum);
            optimizerExpr = weights - lr * wUpt;
        }

        readonly double lr, momentum;
        readonly NullaryExpr<U> wUpt;
        readonly Expr<U> wUptExpr;

        public override BaseOptimizer<U> Clone() => new SGD<U>(lr, momentum);

        public override NDarray<U> Update(NDarray<U> w, NDarray<U> g)
        {
            if (wUpt.Result.GetContent == null)
                wUpt.SetContent(w);

            weights.SetContent(w);
            grad.SetContent(g);
            wUpt.SetContent(wUptExpr.Evaluate());
            return optimizerExpr.Evaluate();
        }
    }
}
