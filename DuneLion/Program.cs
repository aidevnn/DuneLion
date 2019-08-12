using System;
using System.Diagnostics;
using DuneLion.Accuracies;
using DuneLion.Activations;
using DuneLion.Layers;
using DuneLion.Losses;
using DuneLion.Optimizers;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion
{
    class MainClass
    {
        static void Test1()
        {
            Utils.DebugNumpy = Utils.DbgLvlAll;

            var a0 = ND.Uniform(0, 10, 4, 6);
            Console.WriteLine(a0);
            var a1 = ND.Uniform(0, 10, 4, 6);
            Console.WriteLine(a1);
            var a = Variable.CreateNDarray<int>("a");

            var ex = a * 3 + 2;
            Console.WriteLine(ex.Evaluate(("a", a0)));
            Console.WriteLine(ex.Evaluate(("a", a1)));
        }

        static void TestXor<U>(bool summary = false)
        {
            Console.WriteLine($"Hello World! Xor MLP. Backend NDarray<{typeof(Type).Name}>");

            Utils.DebugNumpy = Utils.DbgNo;

            var Xdata = ND.CreateNDarray(new double[4, 2] { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } }).Cast<U>();
            var Ydata = ND.CreateNDarray(new double[4, 1] { { 0 }, { 1 }, { 1 }, { 0 } }).Cast<U>();

            var net = new Network<U>(new SGD<U>(lr: 0.1), new CrossEntropyLoss<U>(), new RoundAccuracy<U>());
            net.AddLayer(new DenseLayer<U>(8, 2));
            net.AddLayer(new TanhLayer<U>());
            net.AddLayer(new DenseLayer<U>(1));
            net.AddLayer(new SigmoidLayer<U>());

            if (summary)
                net.Summary();

            var sw = Stopwatch.StartNew();
            net.Fit(Xdata, Ydata, epochs: 1000, batchSize: 4, displayEpochs: 500);
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");

            Console.WriteLine("Prediction");
            NDarray<U> pred = ND.Round(net.Predict(Xdata).ToExpr(), 6);
            for (int k = 0; k < Xdata.Shape[0]; ++k)
            {
                Console.WriteLine($"[{Xdata.DataAtIdx(k).Glue()}] = [{Ydata.DataAtIdx(k).Glue()}] -> {pred.DataAtIdx(k).Glue()}");
            }
            Console.WriteLine();
        }

        static void TestIris<U>(bool summary = false)
        {
            Console.WriteLine($"Hello World! Iris MLP. Backend NDarray<{typeof(Type).Name}>");

            Utils.DebugNumpy = Utils.DbgNo;
            (var trainX, var trainY, var testX, var testY) = ImportData.IrisDataset<U>(ratio: 0.8);
            Console.WriteLine($"Train on {trainX.Shape[0]}; Test on {testX.Shape[0]}");

            var net = new Network<U>(new SGD<U>(lr: 0.05), new CrossEntropyLoss<U>(), new ArgmaxAccuracy<U>());
            net.AddLayer(new DenseLayer<U>(5, 4));
            net.AddLayer(new TanhLayer<U>());
            net.AddLayer(new DenseLayer<U>(3));
            net.AddLayer(new SigmoidLayer<U>());

            if (summary)
                net.Summary();

            var sw = Stopwatch.StartNew();
            net.Fit(trainX, trainY, epochs: 50, batchSize: 10, displayEpochs: 25);
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");

            net.Test(testX, testY);
            Console.WriteLine();
        }

        static void TestDigits<U>(bool summary = false)
        {
            Console.WriteLine($"Hello World! Digits MLP. Backend NDarray<{typeof(Type).Name}>");

            Utils.DebugNumpy = Utils.DbgNo;
            (var trainX, var trainY, var testX, var testY) = ImportData.DigitsDataset<U>(ratio: 0.9);
            Console.WriteLine($"Train on {trainX.Shape[0]}; Test on {testX.Shape[0]}");

            var net = new Network<U>(new SGD<U>(lr: 0.025), new SquareLoss<U>(), new ArgmaxAccuracy<U>());
            net.AddLayer(new DenseLayer<U>(32, 64));
            net.AddLayer(new TanhLayer<U>());
            net.AddLayer(new DenseLayer<U>(10));
            net.AddLayer(new SigmoidLayer<U>());

            if (summary)
                net.Summary();

            var sw = Stopwatch.StartNew();
            net.Fit(trainX, trainY, epochs: 50, batchSize: 90, displayEpochs: 5);
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");

            net.Test(testX, testY);
            Console.WriteLine();
        }

        public static void Main(string[] args)
        {
            //TestXor<double>();
            //TestIris<double>();
            //TestDigits<double>();

            int N = 5;
            for (int k = 0; k < N; ++k)
                TestIris<double>();

        }
    }
}
