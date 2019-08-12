using System;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Accuracies
{
    public class RoundAccuracy<U> : BaseAccuracy<U>
    {
        public RoundAccuracy()
        {
            Name = "RoundAccuracy";
            accExpr = ND.Eq(ND.Round(y0), ND.Round(p0)).Prod(1).Mean(0).MeanAll();
        }

        public override NDarray<double> Acc(NDarray<U> y, NDarray<U> p) => accExpr.Evaluate(("y", y), ("p", p));
    }
}
