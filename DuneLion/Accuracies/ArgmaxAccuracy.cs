using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Accuracies
{
    public class ArgmaxAccuracy<U> : BaseAccuracy<U>
    {
        public ArgmaxAccuracy()
        {
            Name = "ArgmaxAccuracy";
            accExpr = ND.Eq(ND.ArgMax(y0, 1), ND.ArgMax(p0, 1)).Mean(0).MeanAll();
        }

        public override NDarray<double> Acc(NDarray<U> y, NDarray<U> p) => accExpr.Evaluate(("y", y), ("p", p));
    }
}
