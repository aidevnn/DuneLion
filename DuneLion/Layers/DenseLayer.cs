using System;
using System.Linq;
using DuneLion.Activations;
using DuneLion.Optimizers;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Layers
{
    public class DenseLayer<U> : BaseLayer<U>
    {
        public DenseLayer(int nodes)
        {
            OutputShape = new int[] { nodes };
            Name = "DenseLayer";
        }

        public DenseLayer(int nodes, int inputNodes) : this(nodes)
        {
            InputShape = new int[] { inputNodes };
        }

        NullaryExpr<U> weights, biases, wT;
        BaseOptimizer<U> wOpt, bOpt;
        Expr<U> fwExpr, bwExpr, gwExpr, gbExpr;

        int paramsCount = 0;
        public override int Params => paramsCount;

        public override NDarray<U> Backward(NDarray<U> accumGrad)
        {
            wT.SetContent(weights.T.Evaluate());
            agBw.SetContent(accumGrad);

            if (IsTraining)
            {
                var gw = gwExpr.Evaluate();
                var gb = gbExpr.Evaluate();
                weights.SetContent(wOpt.Update(weights.Evaluate(), gw));
                biases.SetContent(bOpt.Update(biases.Evaluate(), gb));
            }

            return bwExpr.Evaluate();
        }

        public override NDarray<U> Forward(NDarray<U> X, bool isTraining)
        {
            layInp.SetContent(X.Copy);
            return fwExpr.Evaluate(("X", X));
        }
        public override void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
        }

        public override void Initialize(BaseOptimizer<U> optimizer)
        {
            wOpt = optimizer.Clone();
            bOpt = optimizer.Clone();


            double lim = 3.0 / Math.Sqrt(InputShape[0]);

            var w0 = ND.Uniform(-lim, lim, InputShape[0], OutputShape[0]).Cast<U>();
            var b0 = ND.Zeros<U>(1, OutputShape[0]);
            paramsCount = w0.Count + b0.Count;

            weights = Variable.CreateNDarray<U>("w", w0);
            biases = Variable.CreateNDarray<U>("b", b0);
            wT = Variable.CreateNDarray<U>("wT");

            fwExpr = ND.Dot(xFw, weights) + biases;

            gwExpr = ND.Dot(layInp.T, agBw);
            gbExpr = agBw.Sum(0, true);
            bwExpr = ND.Dot(agBw, wT);
        }
    }
}
