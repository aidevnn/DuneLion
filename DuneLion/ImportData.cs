using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NDarrayLib;
using NDarrayLib.Expressions;

namespace DuneLion
{
    public static class ImportData
    {
        public static (NDarray<U>, NDarray<U>, NDarray<U>, NDarray<U>) DigitsDataset<U>(double ratio)
        {
            Func<int, IEnumerable<double>> func0 = v => Enumerable.Range(0, 10).Select(v0 => v == v0 ? 1.0 : 0.0);
            Func<double, int, IEnumerable<double>> func1 = (v, i) => i % 65 != 64 ? new double[] { v } : func0((int)v);

            var raw = File.ReadAllLines("datasets/digits.csv").ToArray();

            var data = raw.SelectMany(l => l.Split(',')).Select(double.Parse).ToArray();
            data = data.SelectMany(func1).ToArray();

            var nDarray = ND.CreateNDarray(data: data, shape: new int[] { -1, 74 });
            var idx0 = (int)(nDarray.Shape[0] * ratio);
            (var train, var test) = ND.Split(nDarray, axis: 0, idx: idx0);

            (var trainX, var trainY) = ND.Split(train, axis: 1, idx: 64);
            (var testX, var testY) = ND.Split(test, axis: 1, idx: 64);

            trainX = trainX.ToExpr() / 16.0;
            testX = testX.ToExpr() / 16.0;

            return (trainX.Cast<U>(), trainY.Cast<U>(), testX.Cast<U>(), testY.Cast<U>());
        }

        public static (NDarray<U>, NDarray<U>, NDarray<U>, NDarray<U>) IrisDataset<U>(double ratio)
        {
            var raw = File.ReadAllLines("datasets/iris.csv").ToArray();
            var data = raw.SelectMany(l => l.Split(',')).Select(double.Parse).ToArray();

            var nDarray = ND.CreateNDarray(data: data, shape: new int[] { -1, 7 });
            var idx0 = (int)(nDarray.Shape[0] * ratio);
            (var train, var test) = ND.Split(nDarray, axis: 0, idx: idx0);

            (var trainX, var trainY) = ND.Split(train, axis: 1, idx: 4);
            (var testX, var testY) = ND.Split(test, axis: 1, idx: 4);

            var vmax = ND.Max(ND.MaxAxis(trainX.ToExpr(), axis: 0, keepdims: true), ND.MaxAxis(testX.ToExpr(), axis: 0, keepdims: true));
            trainX = trainX.ToExpr() / vmax;
            testX = testX.ToExpr() / vmax;

            return (trainX.Cast<U>(), trainY.Cast<U>(), testX.Cast<U>(), testY.Cast<U>());
        }
    }
}
