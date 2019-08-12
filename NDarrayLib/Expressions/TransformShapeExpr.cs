using System;
using System.Text;

namespace NDarrayLib.Expressions
{
    public enum TransformType { Reshape, Transpose, Pad }

    public class TransformShapeExpr<U> : Expr<U>
    {
        public TransformShapeExpr(TransformType transformType, Expr<U> expr, int[] arr)
        {
            foreach (var e in expr.Parameters)
                if (!Parameters.ContainsKey(e.Key))
                    Parameters[e.Key] = e.Value;

            Name = $"{transformType} ({arr.Glue()})";
            this.expr = expr;
            this.arr = arr;
            this.transformType = transformType;
            this.expr.IsRoot = false;
        }

        readonly Expr<U> expr;
        readonly int[] arr;
        readonly TransformType transformType;

        public override string Display(string indent = "")
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"{indent}{Name}");
            sb.Append(expr.Display(indent + "|___"));
            return sb.ToString();
        }

        public override void Eval()
        {
            expr.Eval();
            if (transformType == TransformType.Reshape)
                ND.ApplyReshape(expr, arr, this);
            else if (transformType == TransformType.Transpose)
                ND.ApplyTranspose(expr, arr, this);
        }
    }
}
