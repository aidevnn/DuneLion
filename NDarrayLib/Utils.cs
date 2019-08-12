using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib
{
    public static class Utils
    {
        public static string Glue<T>(this IEnumerable<T> ts, string sep = " ", string format = "{0}") =>
            string.Join(sep, ts.Select(a => string.Format(format, a)));

        public const int DbgNo = 0, DbgLvl1 = 0b1, DbgLvl2 = 0b10, DbgLvlAll = 0b11;
        public static int DebugNumpy = DbgNo;

        public static int NbDecimales = 8;
        public static bool IsDebugNo => DebugNumpy == DbgNo;
        public static bool IsDebugLvl1 => (DebugNumpy & DbgLvl1) == DbgLvl1;
        public static bool IsDebugLvl2 => (DebugNumpy & DbgLvl2) == DbgLvl2;

        //private static readonly Random random = new Random(123);
        private static readonly Random random = new Random((int)DateTime.Now.Ticks);
        public static Random GetRandom => random;

        public static int ArrMul(int[] shape, int start = 0) => shape.Skip(start).Aggregate(1, (a, i) => a * i);
        public static int[] Shape2Strides(int[] shape) => Enumerable.Range(0, shape.Length).Select(i => ArrMul(shape, i + 1)).ToArray();

        public static int Array2IntIndex(int[] args, int[] shape, int[] strides)
        {
            int idx = 0;
            for (int k = 0; k < args.Length; ++k)
            {
                var v = args[k];
                idx += v * strides[k];
            }

            return idx;
        }

        public static void Int2ArrayIndex(int idx, int[] shape, int[] indices)
        {

            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                indices[k] = idx % sk;
                idx = idx / sk;
            }
        }

        public static int Int2IntIndex(int idx0, int[] shape, int[] strides)
        {
            int idx1 = 0;
            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                idx1 += strides[k] * (idx0 % sk);
                idx0 = idx0 / sk;
            }

            return idx1;
        }

        public static int[] PrepareReshape(int[] baseShape, int[] shape) => PrepareReshape(ArrMul(baseShape), shape);

        public static int[] PrepareReshape(int dim0, int[] shape)
        {
            int mone = shape.Count(i => i == -1);
            if (mone > 1)
                throw new ArgumentException("Can only specify one unknown dimension");

            if (mone == 1)
            {
                int idx = shape.ToList().FindIndex(i => i == -1);
                shape[idx] = 1;
                var dim2 = ArrMul(shape);
                shape[idx] = dim0 / dim2;
            }

            var dim1 = ArrMul(shape);

            if (dim0 != dim1)
                throw new ArgumentException($"cannot reshape array of size {dim0} into shape ({shape.Glue()})");

            return shape;
        }

        public static int[] PrepareTranspose(int rank) => Enumerable.Range(0, rank).Reverse().ToArray();
        public static int[] DoTranspose(int[] arr, int[] table) => Enumerable.Range(0, arr.Length).Select(i => arr[table[i]]).ToArray();

        public static int[] BroadCastShapes(int[] shape0, int[] shape1)
        {
            int sLength0 = shape0.Length;
            int sLength1 = shape1.Length;
            int mLength = Math.Max(sLength0, sLength1);

            int[] nshape = new int[mLength];
            for (int k = mLength - 1, i = sLength0 - 1, j = sLength1 - 1; k >= 0; --k, --i, --j)
            {
                int idx0 = i < 0 ? 1 : shape0[i];
                int idx1 = j < 0 ? 1 : shape1[j];
                if (idx0 != idx1 && idx0 != 1 && idx1 != 1)
                    throw new ArgumentException($"Cannot broadcast ({shape0.Glue()}) with ({shape1.Glue()})");

                nshape[k] = Math.Max(idx0, idx1);
            }

            return nshape;
        }

        public static bool SameShape(int[] shape0, int[] shape1)
        {
            if (ArrMul(shape0) != ArrMul(shape1))
                return false;

            int sLength0 = shape0.Length;
            int sLength1 = shape1.Length;
            int mLength = Math.Min(sLength0, sLength1);

            for (int k = mLength - 1; k >= 0; --k)
                if (shape0[k] != shape1[k])
                    return false;

            return true;
        }

        public static int[] PrepareAxisOps(int[] shape, int axis, bool keepdims)
        {
            List<int> nshape = new List<int>(shape);

            if (axis == -1)
                nshape = Enumerable.Repeat(1, shape.Length).ToList();
            else
                nshape[axis] = 1;

            if (!keepdims)
            {
                if (axis == -1)
                    nshape = new List<int>() { 1 };
                else
                    nshape.RemoveAt(axis);
            }

            return nshape.ToArray();
        }

        public static int[] PrepareCumSumProd(int[] shape, int axis)
        {
            if (axis < -1 || axis >= shape.Length)
                throw new ArgumentException("Bad axis for CumSumProd");

            if (axis == -1)
                return new int[] { ArrMul(shape) };

            return shape;
        }

        public static (int[], int[]) PrepareArgMinmax(int[] shape, int axis)
        {
            if (axis < 0 || axis >= shape.Length)
                throw new ArgumentException("Bad axis for ArgMinMax");

            var ishape = shape.ToArray();
            ishape[axis] = 1;
            var nshape = shape.Select((v, i) => (v, i)).Where(t => t.i != axis).Select(t => t.v).ToArray();
            return (ishape, nshape);
        }

        public static (int[], int[], int[], int[]) PrepareDot(int[] shape0, int[] shape1)
        {
            bool head = false, tail = false;
            int[] nshape;
            int[] lshape, rshape, idxInfos;

            if (head = shape0.Length == 1)
                lshape = new int[] { 1, shape0[0] };
            else
                lshape = shape0.ToArray();

            if (tail = shape1.Length == 1)
                rshape = new int[] { shape1[0], 1 };
            else
                rshape = shape1.ToArray();


            int length0 = lshape.Length;
            int length1 = rshape.Length;
            int piv = lshape.Last();

            if (piv != rshape[length1 - 2])
                throw new ArgumentException($"Cannot multiply ({shape0.Glue()}) and ({shape1.Glue()})");

            nshape = new int[length0 + length1 - 2];
            idxInfos = new int[length0 + length1 - 2];

            for (int k = 0, k0 = 0; k < length0 + length1; ++k)
            {
                if (k == length0 - 1 || k == length0 + length1 - 2) continue;
                if (k < length0 - 1) nshape[k] = lshape[idxInfos[k] = k];
                else nshape[k0] = rshape[idxInfos[k0] = k - length0];
                ++k0;
            }

            return (lshape, rshape, nshape, idxInfos);
        }

        public static int[] PrepareConcatene(int[] shape0, int[] shape1, int axis)
        {
            if (shape0.Length != shape1.Length)
                throw new ArgumentException($"Cannot concat rank={shape0.Length} and rank={shape1.Length}");

            if (axis < 0 || axis >= shape0.Length)
                throw new ArgumentException("Bad axis concatenation");

            for (int k = 0; k < shape0.Length; ++k)
            {
                if (k == axis) continue;

                if (shape0[k] != shape1[k])
                    throw new ArgumentException($"Cannot concat ({shape0.Glue()}) and ({shape1.Glue()}) along axis={axis}");
            }

            var nshape = shape0.ToArray();
            nshape[axis] += shape1[axis];

            return nshape;
        }

        public static (int[], int[]) PrepareSplit(int[] shape, int axis, int idx)
        {
            if (axis < 0 || axis >= shape.Length)
                throw new ArgumentException("Bad Split axis");

            int dim = shape[axis];
            if (idx < 0 || idx >= dim)
                throw new ArgumentException("Bad Split index");

            int[] shape0 = shape.ToArray();
            int[] shape1 = shape.ToArray();
            shape0[axis] = idx;
            shape1[axis] -= idx;

            return (shape0, shape1);
        }
    }
}
