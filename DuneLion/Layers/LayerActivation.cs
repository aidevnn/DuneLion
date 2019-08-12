using System.Linq;
using DuneLion.Activations;
using DuneLion.Optimizers;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Layers
{
    public class LayerActivation<U> : BaseLayer<U>
    {
        public LayerActivation(BaseActivation<U> activation)
        {
            this.activation = activation;
            Name = activation.Name;

            actDeriv = Variable.CreateNDarray<U>("actDeriv");
            backwardExpr = agBw * actDeriv;
        }

        readonly BaseActivation<U> activation;
        readonly NullaryExpr<U> actDeriv;
        readonly Expr<U> backwardExpr;

        public override int Params => 0;

        public override NDarray<U> Backward(NDarray<U> accumGrad)
        {
            return backwardExpr.Evaluate(("accumGrad", accumGrad),("actDeriv", activation.Derivative(layInp)));
        }

        public override NDarray<U> Forward(NDarray<U> X, bool isTraining)
        {
            layInp.SetContent(X);
            return activation.Func(X);
        }

        public override void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            OutputShape = shape.ToArray();
        }

        public override void Initialize(BaseOptimizer<U> optimizer)
        {

        }
    }

    public class SigmoidLayer<U> : LayerActivation<U>
    {
        public SigmoidLayer() : base(new SigmoidActivation<U>()) { }
    }

    public class TanhLayer<U> : LayerActivation<U>
    {
        public TanhLayer() : base(new TanhActivation<U>()) { }
    }
}
