using System;
using System.Collections.Generic;
using System.Linq;

using DuneLion.Accuracies;
using DuneLion.Layers;
using DuneLion.Losses;
using DuneLion.Optimizers;

using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion
{
    public class Network<U>
    {
        public Network(BaseOptimizer<U> optimizer, BaseLoss<U> lossf, BaseAccuracy<U> accuracy)
        {
            this.optimizer = optimizer;
            this.lossf = lossf;
            this.accuracy = accuracy;
        }

        readonly BaseOptimizer<U> optimizer;
        readonly BaseLoss<U> lossf;
        readonly BaseAccuracy<U> accuracy;

        public List<BaseLayer<U>> layers = new List<BaseLayer<U>>();

        public void SetTrainable(bool train = true) => layers.ForEach(l => l.IsTraining = train);

        public void AddLayer(BaseLayer<U> layer)
        {
            if (layers.Count != 0)
                layer.SetInputShape(layers.Last().OutputShape);

            layer.Initialize(optimizer);
            layers.Add(layer);
        }

        public NDarray<U> ForwardPass(NDarray<U> X, bool isTraining = true)
        {
            var layerOutput = X.Copy;
            foreach (var layer in layers)
                layerOutput = layer.Forward(layerOutput, isTraining);

            return layerOutput;
        }

        public void BackwardPass(NDarray<U> lossGrad)
        {
            foreach (var layer in layers.Reverse<BaseLayer<U>>())
                lossGrad = layer.Backward(lossGrad);
        }

        public NDarray<U> Predict(NDarray<U> X) => ForwardPass(X, false);

        (double, double) TestOnBatch(NDarray<U> X, NDarray<U> y)
        {
            SetTrainable(false);
            var yp = ForwardPass(X, false);
            var loss = lossf.Loss(y, yp).Data[0];
            var acc = accuracy.Acc(y, yp).Data[0];

            return (loss, acc);
        }

        public (double, double) TrainOnBatch(NDarray<U> X, NDarray<U> y)
        {
            var yp = ForwardPass(X);
            var loss = lossf.Loss(y, yp).Data[0];
            var acc = accuracy.Acc(y, yp).Data[0];
            var lossGrad = lossf.Grad(y, yp);
            BackwardPass(lossGrad);

            return (loss, acc);
        }

        public void Summary()
        {
            Console.WriteLine("Summary");
            Console.WriteLine($"Network {optimizer.Name} {lossf.Name} {accuracy.Name}");
            Console.WriteLine($"Input  Shape:{layers[0].InputShape.Glue()}");
            int tot = 0;
            foreach (var layer in layers)
            {
                Console.WriteLine($"Layer: {layer.Name,-20} Parameters: {layer.Params,5} Nodes[In:{layer.InputShape.Glue(),2} -> Out:{layer.OutputShape.Glue()}]");
                tot += layer.Params;
            }

            Console.WriteLine($"Output Shape:{layers.Last().OutputShape.Glue()}");
            Console.WriteLine($"Total Parameters:{tot}");
            Console.WriteLine();
        }

        public void Test(NDarray<U> testX, NDarray<U> testY)
        {
            var (loss, acc) = TestOnBatch(testX, testY);
            Console.WriteLine("TestResult Loss:{0:0.000000} Acc:{1:0.0000}", loss, acc);
        }

        public void Fit(NDarray<U> X, NDarray<U> y, int epochs, int batchSize = 64, int displayEpochs = 1)
        {
            Console.WriteLine("Start Training...");

            SetTrainable();

            for (int k = 0; k <= epochs; ++k)
            {
                List<double> losses = new List<double>();
                List<double> accs = new List<double>();

                var batchData = ND.BatchIterator(X, y, batchSize);
                foreach (var batch in batchData)
                {
                    var (loss, acc) = TrainOnBatch(batch.Item1, batch.Item2);
                    losses.Add(loss);
                    accs.Add(acc);
                }

                if (k % displayEpochs == 0)
                    Console.WriteLine("Epochs {0,5}/{1} Loss:{2:0.000000} Acc:{3:0.0000}", k, epochs, losses.Average(), accs.Average());
            }
            Console.WriteLine("End Training.");
        }
    }
}
