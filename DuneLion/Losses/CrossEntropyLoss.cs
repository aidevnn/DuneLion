using System;
using NDarrayLib;
using NDarrayLib.Expressions;
namespace DuneLion.Losses
{
    public class CrossEntropyLoss<U> : BaseLoss<U>
    {
        public CrossEntropyLoss()
        {
            Name = "CrossEntropyLoss";
            pc = ND.Clamp(p0, 1e-12, 1 - 1e-12);
            lossExpr = (-y0 * ND.Log(pc) - (1 - y0) * ND.Log(1 - pc)).MeanAll();
            gradExpr = -y0 / pc + (1 - y0) / (1 - pc);
        }

        readonly Expr<U> pc;

        public override NDarray<U> Grad(NDarray<U> y, NDarray<U> p)
        {
            y0.SetContent(y);
            p0.SetContent(p);
            pc.Evaluate();
            //return gradExpr;
            return -y0 / pc + (1 - y0) / (1 - pc);
        }

        public override NDarray<double> Loss(NDarray<U> y, NDarray<U> p)
        {
            y0.SetContent(y);
            p0.SetContent(p);
            pc.Evaluate();
            //return lossExpr;
            return (-y0 * ND.Log(pc) - (1 - y0) * ND.Log(1 - pc)).MeanAll();
        }
    }
}
