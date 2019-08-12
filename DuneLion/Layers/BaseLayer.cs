using System;
using DuneLion.Optimizers;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion.Layers
{
    public abstract class BaseLayer<U>
    {
        public string Name { get; protected set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }
        public bool IsTraining { get; set; }

        public abstract int Params { get; }

        protected BaseLayer()
        {
            xFw = Variable.CreateNDarray<U>("X");
            agBw = Variable.CreateNDarray<U>("accumGrad");
            layInp = Variable.CreateNDarray<U>("layerInput");
        }

        protected readonly NullaryExpr<U> xFw, agBw, layInp;

        public abstract void Initialize(BaseOptimizer<U> optimizer);

        public abstract void SetInputShape(int[] shape);
        public abstract NDarray<U> Forward(NDarray<U> X, bool isTraining);
        public abstract NDarray<U> Backward(NDarray<U> accumGrad);
    }
}
