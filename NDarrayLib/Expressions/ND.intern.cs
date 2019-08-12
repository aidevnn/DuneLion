using System;
using System.Linq;

namespace NDarrayLib.Expressions
{
    public static partial class ND
    {
        public static void ApplyReshape<U>(Expr<U> inExpr, int[] shape, Expr<U> outExpr)
        {
            NDarray<U> inArr = inExpr.Result.GetContent as NDarray<U>;
            if (!(outExpr.Result?.GetContent is NDarray<U> outArr))
            {
                var nshape = Utils.PrepareReshape(inArr.Count, shape);
                outArr = new NDarray<U>(data: inArr.Data, shape: nshape);
                //outExpr.Result = Variable.CreateNDarray(outArr).Result;
                outExpr.SetContent(outArr);
                return;
            }

            if (inArr.Count != outArr.Count)
                throw new ArgumentException($"Cannot reshape {inArr.Count} to ({outArr.Shape.Glue()})");

            for (int idx = 0; idx < outArr.Count; ++idx)
                outArr.Data[idx] = inArr.Data[idx];
        }

        public static void ApplyTranspose<U>(Expr<U> inExpr, int[] table, Expr<U> outExpr)
        {
            NDarray<U> inArr = inExpr.Result.GetContent as NDarray<U>;
            if (table == null || table.Length == 0)
                table = Utils.PrepareTranspose(inArr.Shape.Length);

            var nshape = Utils.DoTranspose(inArr.Shape, table);
            var nstrides = Utils.DoTranspose(inArr.Strides, table);

            if (!(outExpr.Result?.GetContent is NDarray<U> outArr))
            {
                outArr = new NDarray<U>(shape: nshape);
                //outExpr.Result = Variable.CreateNDarray(outArr).Result;
                outExpr.SetContent(outArr);
            }

            if (inArr.Count != outArr.Count)
                throw new ArgumentException("NDarray result has bad shape");

            for (int idx = 0; idx < outArr.Count; ++idx)
            {
                int idx1 = Utils.Int2IntIndex(idx, nshape, nstrides);
                outArr.Data[idx] = inArr.Data[idx1];
            }
        }

        public static void ApplyFunc<U, V>(Expr<U> inExpr, Func<U, V> func, Expr<V> outExpr)
        {
            NDarray<U> inArr = inExpr.Result.GetContent as NDarray<U>;
            if (!(outExpr.Result?.GetContent is NDarray<V> outArr))
            {
                outArr = new NDarray<V>(shape: inArr.Shape, strides: inArr.Strides);
                //outExpr.Result = Variable.CreateNDarray(outArr).Result;
                outExpr.SetContent(outArr);
            }

            if (inArr.Count != outArr.Count)
                throw new ArgumentException("NDarray result has bad shape");

            for (int idx = 0; idx < outArr.Count; ++idx)
                outArr.Data[idx] = func(inArr.Data[idx]);
        }

        public static void ApplyElementWiseOp<U, V>(Expr<U> leftExpr, Expr<U> rightExpr, Func<U, U, V> func, Expr<V> outExpr)
        {
            var left = leftExpr.Result.GetContent as NDarray<U>;
            var right = rightExpr.Result.GetContent as NDarray<U>;
            if (!(outExpr.Result?.GetContent is NDarray<V> outArr))
            {
                var nshape = Utils.BroadCastShapes(left.Shape, right.Shape);
                outArr = new NDarray<V>(shape: nshape);
                //outExpr.Result = Variable.CreateNDarray(outArr).Result;
                outExpr.SetContent(outArr);
            }

            if (Utils.SameShape(left.Shape, right.Shape))
            {
                for (int idx = 0; idx < outArr.Count; ++idx)
                    outArr.Data[idx] = func(left.Data[idx], right.Data[idx]);
                return;
            }

            for (int index = 0; index < outArr.Count; ++index)
            {
                Utils.Int2ArrayIndex(index, outArr.Shape, outArr.Indices);
                for (int k = outArr.Indices.Length - 1, i = left.Shape.Length - 1, j = right.Shape.Length - 1; k >= 0; --k, --i, --j)
                {
                    if (i >= 0) left.Indices[i] = outArr.Indices[k] % left.Shape[i];
                    if (j >= 0) right.Indices[j] = outArr.Indices[k] % right.Shape[j];
                }

                var v0 = left.Data[Utils.Array2IntIndex(left.Indices, left.Shape, left.Strides)];
                var v1 = right.Data[Utils.Array2IntIndex(right.Indices, right.Shape, right.Strides)];
                outArr.Data[index] = func(v0, v1);
            }
        }

        public static void ApplyAxisOps<U>(Expr<U> inExpr, Expr<U> outExpr, int axis, bool keepdims, Func<U, U, U> func, U start, bool mean = false)
        {
            var inArr = inExpr.Result.GetContent as NDarray<U>;
            if (!(outExpr.Result?.GetContent is NDarray<U> outArr))
            {
                var shape = Utils.PrepareAxisOps(inArr.Shape, axis, keepdims);
                outArr = new NDarray<U>(shape: shape);
                //outExpr.Result = Variable.CreateNDarray(outArr).Result;
                outExpr.SetContent(outArr);
            }

            if (axis == -1)
            {
                U res = start;
                U nb = mean ? NDarray<U>.OpsT.Cast(inArr.Count) : NDarray<U>.OpsT.One;
                for (int idx = 0; idx < inArr.Count; ++idx)
                    res = func(res, inArr.Data[idx]);

                res = NDarray<U>.OpsT.Div(res, nb);
                outArr.Data[0] = res;
            }
            else
            {
                var NShape = Utils.PrepareAxisOps(inArr.Shape, axis, true);
                var NIndices = new int[NShape.Length];
                U nb = mean ? NDarray<U>.OpsT.Cast(inArr.Shape[axis]) : NDarray<U>.OpsT.One;

                for (int idx0 = 0; idx0 < outArr.Count; ++idx0)
                {
                    U res = start;
                    Utils.Int2ArrayIndex(idx0, NShape, NIndices);

                    for (int k = 0; k < inArr.Shape[axis]; ++k)
                    {
                        NIndices[axis] = k;
                        int idx1 = Utils.Array2IntIndex(NIndices, inArr.Shape, inArr.Strides);
                        res = func(res, inArr.Data[idx1]);
                    }

                    outArr.Data[idx0] = NDarray<U>.OpsT.Div(res, nb);
                }
            }
        }


        public static void ApplyArgMinMax<U>(Expr<U> inExpr, Expr<int> outExpr, int axis, Func<U, U, U> func, U tmp)
        {
            var inArr = inExpr.Result.GetContent as NDarray<U>;
            (int[] ishape, int[] nshape) = Utils.PrepareArgMinmax(inArr.Shape, axis);
            int[] indices = new int[ishape.Length];
            if (!(outExpr.Result?.GetContent is NDarray<int> outArr))
            {
                outArr = new NDarray<int>(nshape);
                //outExpr.Result = Variable.CreateNDarray(outArr).Result;
                outExpr.SetContent(outArr);
            }

            int nb = inArr.Shape[axis];
            for(int idx = 0; idx < outArr.Count; ++idx)
            {
                U valBest = tmp;
                int idxBest = 0;
                Utils.Int2ArrayIndex(idx, ishape, indices);
                for (int k = 0; k < nb; ++k)
                {
                    indices[axis] = k;
                    var v = inArr.Data[Utils.Array2IntIndex(indices, inArr.Shape, inArr.Strides)];
                    var v0 = func(v, valBest);
                    if (!valBest.Equals(v0))
                    {
                        idxBest = k;
                        valBest = v0;
                    }
                }

                outArr.Data[idx] = idxBest;
            }
        }

        public static void ApplyDotExpr<U>(Expr<U> leftExpr, Expr<U> rightExpr, Expr<U> outExpr)
        {
            var left = leftExpr.Result.GetContent as NDarray<U>;
            var right = rightExpr.Result.GetContent as NDarray<U>;
            (int[] lshape, int[] rshape, int[] shape, int[] idxInfos) = Utils.PrepareDot(left.Shape, right.Shape);
            if (!(outExpr.Result?.GetContent is NDarray<U> outArr))
            {
                outArr = new NDarray<U>(shape: shape);
                //outExpr.Result = Variable.CreateNDarray(outArr).Result;
                outExpr.SetContent(outArr);
            }

            var leftArr = left.Shape.Length == lshape.Length ? left : new NDarray<U>(data: left.Data, shape: lshape);
            var rightArr = right.Shape.Length == rshape.Length ? right : new NDarray<U>(data: right.Data, shape: rshape);

            int length0 = lshape.Length;
            int length1 = rshape.Length;
            int piv = lshape.Last();

            int[] indices = new int[shape.Length];
            for (int idx = 0; idx < outArr.Count; ++idx)
            {
                U sum = NDarray<U>.OpsT.Zero;
                Utils.Int2ArrayIndex(idx, shape, indices);

                for (int k = 0; k < shape.Length; ++k)
                {
                    if (k < length0 - 1) leftArr.Indices[idxInfos[k]] = indices[k];
                    else rightArr.Indices[idxInfos[k]] = indices[k];
                }

                for (int i = 0; i < piv; ++i)
                {
                    leftArr.Indices[length0 - 1] = rightArr.Indices[length1 - 2] = i;

                    int idxl = Utils.Array2IntIndex(leftArr.Indices, leftArr.Shape, leftArr.Strides);
                    int idxr = Utils.Array2IntIndex(rightArr.Indices, rightArr.Shape, rightArr.Strides);
                    var prod = NDarray<U>.OpsT.Mul(leftArr.Data[idxl], rightArr.Data[idxr]);
                    sum = NDarray<U>.OpsT.Add(sum, prod);
                }

                outArr.Data[idx] = sum;
            }
        }
    }
}
