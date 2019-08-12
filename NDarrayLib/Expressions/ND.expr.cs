using System;
namespace NDarrayLib.Expressions
{
    public static partial class ND
    {
        public static Expr<U> Abs<U>(Expr<U> expr) => new MathExpr<U>(FuncName.Abs, expr);
        public static Expr<U> Exp<U>(Expr<U> expr) => new MathExpr<U>(FuncName.Exp, expr);
        public static Expr<U> Log<U>(Expr<U> expr) => new MathExpr<U>(FuncName.Log, expr);
        public static Expr<U> Neg<U>(Expr<U> expr) => new MathExpr<U>(FuncName.Neg, expr);
        public static Expr<U> Sq<U>(Expr<U> expr) => new MathExpr<U>(FuncName.Sq, expr);
        public static Expr<U> Sqrt<U>(Expr<U> expr) => new MathExpr<U>(FuncName.Sqrt, expr);
        public static Expr<U> Sigmoid<U>(Expr<U> expr) => new MathExpr<U>(FuncName.Sigmoid, expr);
        public static Expr<U> DSigmoid<U>(Expr<U> expr) => new MathExpr<U>(FuncName.DSigmoid, expr);
        public static Expr<U> Tanh<U>(Expr<U> expr) => new MathExpr<U>(FuncName.Tanh, expr);
        public static Expr<U> DTanh<U>(Expr<U> expr) => new MathExpr<U>(FuncName.DTanh, expr);

        public static Expr<V> DoCast<U, V>(Expr<U> expr) => new ApplyExpr<U, V>($"Cast to {typeof(V).Name}", expr, NDarray<V>.OpsT.Cast);
        public static Expr<U> Round<U>(Expr<U> expr, int dec = 0) => new ApplyExpr<U, U>($"Round {dec}", expr, x => NDarray<U>.OpsT.Round(x, dec));
        public static Expr<U> Clamp<U>(Expr<U> expr, double min, double max) => new ApplyExpr<U, U>($"Clamp [{min},{max}]", expr, x => NDarray<U>.OpsT.Clamp(x, min, max));

        public static Expr<U> Add<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, U>(left, NDarray<U>.OpsT.Add, right);
        public static Expr<U> Sub<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, U>(left, NDarray<U>.OpsT.Sub, right);
        public static Expr<U> Mul<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, U>(left, NDarray<U>.OpsT.Mul, right);
        public static Expr<U> Div<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, U>(left, NDarray<U>.OpsT.Div, right);
        public static Expr<U> Min<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, U>(left, NDarray<U>.OpsT.Min, right);
        public static Expr<U> Max<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, U>(left, NDarray<U>.OpsT.Max, right);

        public static Expr<double> Eq<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, double>(left, NDarray<U>.OpsT.Eq, right);
        public static Expr<double> Neq<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, double>(left, NDarray<U>.OpsT.Neq, right);
        public static Expr<double> Lt<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, double>(left, NDarray<U>.OpsT.Lt, right);
        public static Expr<double> Gt<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, double>(left, NDarray<U>.OpsT.Gt, right);
        public static Expr<double> Lte<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, double>(left, NDarray<U>.OpsT.Lte, right);
        public static Expr<double> Gte<U>(Expr<U> left, Expr<U> right) => new ElementWiseExpr<U, double>(left, NDarray<U>.OpsT.Gte, right);

        public static Expr<U> Reshape<U>(Expr<U> expr, params int[] shape) => new TransformShapeExpr<U>(TransformType.Reshape, expr, shape);
        public static Expr<U> Transpose<U>(Expr<U> expr, params int[] table) => new TransformShapeExpr<U>(TransformType.Transpose, expr, table);

        public static Expr<U> SumAxis<U>(Expr<U> expr, int axis = -1, bool keepdims = false) => new AxisOpsExpr<U>(AxisOpsType.Sum, expr, axis, keepdims);
        public static Expr<U> ProdAxis<U>(Expr<U> expr, int axis = -1, bool keepdims = false) => new AxisOpsExpr<U>(AxisOpsType.Prod, expr, axis, keepdims);
        public static Expr<U> MeanAxis<U>(Expr<U> expr, int axis = -1, bool keepdims = false) => new AxisOpsExpr<U>(AxisOpsType.Mean, expr, axis, keepdims);
        public static Expr<U> MinAxis<U>(Expr<U> expr, int axis = -1, bool keepdims = false) => new AxisOpsExpr<U>(AxisOpsType.Min, expr, axis, keepdims);
        public static Expr<U> MaxAxis<U>(Expr<U> expr, int axis = -1, bool keepdims = false) => new AxisOpsExpr<U>(AxisOpsType.Max, expr, axis, keepdims);

        public static Expr<int> ArgMin<U>(Expr<U> expr, int axis) => new ArgMinMaxExpr<U>(ArgIndex.Min, axis, expr);
        public static Expr<int> ArgMax<U>(Expr<U> expr, int axis) => new ArgMinMaxExpr<U>(ArgIndex.Max, axis, expr);

        public static Expr<U> Dot<U>(Expr<U> left, Expr<U> right) => new DotExpr<U>(left, right);

        public static Expr<U> ToExpr<U>(this NDarray<U> nDarray) => Variable.CreateNDarray(nDarray);

    }
}
