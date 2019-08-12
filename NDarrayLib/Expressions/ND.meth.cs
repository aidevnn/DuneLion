using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib.Expressions
{
    public static partial class ND
    {
        public static NDarray<U> Scalar<U>(U v, params int[] shape) => new NDarray<U>(v0: v, shape: shape);
        public static NDarray<U> Zeros<U>(params int[] shape) => Scalar(NDarray<U>.OpsT.Zero, shape);
        public static NDarray<U> Ones<U>(params int[] shape) => Scalar(NDarray<U>.OpsT.One, shape);

        public static NDarray<int> Arange(int start, int length, int step = 1)
        {
            int[] data = Enumerable.Range(0, length).Select(i => start + i * step).ToArray();
            return new NDarray<int>(data: data, shape: new int[] { length });
        }

        public static NDarray<int> Arange(int length) => Arange(0, length, 1);

        public static NDarray<U> Uniform<U>(U min, U max, params int[] shape)
        {
            int count = Utils.ArrMul(shape);
            U[] data = Enumerable.Range(0, count).Select(i => NDarray<U>.OpsT.Rand(min, max)).ToArray();
            return new NDarray<U>(data: data, shape: shape);
        }

        public static NDarray<U> CreateNDarray<U>(U[] data, params int[] shape)
            => new NDarray<U>(data: data, shape: Utils.PrepareReshape(data.Length, shape));

        public static NDarray<U> CreateNDarray<U>(U[,] data)
        {
            int dim0 = data.GetLength(0);
            int dim1 = data.GetLength(1);

            U[] data0 = new U[dim0 * dim1];
            for (int i = 0; i < dim0; ++i)
                for (int j = 0; j < dim1; ++j)
                    data0[i * dim1 + j] = data[i, j];

            return CreateNDarray(data0, new int[] { dim0, dim1 });
        }

        public static (NDarray<U>, NDarray<U>) Split<U>(NDarray<U> nDarray, int axis, int idx)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine("Split");

            (int[] nshape0, int[] nshape1) = Utils.PrepareSplit(nDarray.Shape, axis, idx);
            var nd0 = new NDarray<U>(nshape0);
            var nd1 = new NDarray<U>(nshape1);

            for (int idx0 = 0; idx0 < nd0.Count; ++idx0)
            {
                Utils.Int2ArrayIndex(idx0, nd0.Shape, nDarray.Indices);
                var idx2 = Utils.Array2IntIndex(nDarray.Indices, nDarray.Shape, nDarray.Strides);
                nd0.Data[idx0] = nDarray.Data[idx2];
            }

            for (int idx1 = 0; idx1 < nd1.Count; ++idx1)
            {
                Utils.Int2ArrayIndex(idx1, nd1.Shape, nDarray.Indices);
                nDarray.Indices[axis] += idx;
                var idx2 = Utils.Array2IntIndex(nDarray.Indices, nDarray.Shape, nDarray.Strides);
                nd1.Data[idx1] = nDarray.Data[idx2];
            }

            return (nd0, nd1);
        }

        public static List<(NDarray<Type>, NDarray<Type>)> BatchIterator<Type>(NDarray<Type> X, NDarray<Type> Y, int batchsize = 64, bool shuffle = true)
        {
            int dim0 = X.Shape[0];
            if (Y.Shape[0] != dim0)
                throw new ArgumentException();

            if (batchsize > dim0)
                batchsize = dim0;

            List<(NDarray<Type>, NDarray<Type>)> allBatch = new List<(NDarray<Type>, NDarray<Type>)>();
            int nb = dim0 / batchsize;

            var ltIdx = new Queue<int>(Enumerable.Range(0, dim0));
            if (shuffle)
                ltIdx = new Queue<int>(Enumerable.Range(0, dim0).OrderBy(t => Utils.GetRandom.NextDouble()));

            var xshape = X.Shape.ToArray();
            var yshape = Y.Shape.ToArray();
            xshape[0] = batchsize;
            yshape[0] = batchsize;

            for (int k = 0; k < nb; ++k)
            {
                var xarr = new NDarray<Type>(xshape);
                var yarr = new NDarray<Type>(yshape);
                for (int i = 0; i < batchsize; ++i)
                {
                    int idx = ltIdx.Dequeue();
                    X.DataAtIdx(idx).CopyTo(xarr.Data, i * xarr.Strides[0]);
                    Y.DataAtIdx(idx).CopyTo(yarr.Data, i * yarr.Strides[0]);
                }

                allBatch.Add((xarr, yarr));
            }

            return allBatch;
        }

    }
}
