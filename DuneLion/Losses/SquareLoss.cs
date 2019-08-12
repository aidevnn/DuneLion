using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Losses
{
    public class SquareLoss<U> : BaseLoss<U>
    {
        public SquareLoss()
        {
            Name = "SquareLoss";
            lossExpr = (ND.Sq(p0 - y0) / 2).MeanAll();
            gradExpr = (p0 - y0);
        }

        public override NDarray<U> Grad(NDarray<U> y, NDarray<U> p) => gradExpr.Evaluate(("y", y), ("p", p));

        public override NDarray<double> Loss(NDarray<U> y, NDarray<U> p) => lossExpr.Evaluate(("y", y), ("p", p));
    }
}
