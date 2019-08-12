using System;
using System.Text;

namespace NDarrayLib.Expressions
{
    public enum AxisOpsType { Sum, Prod, Mean, Min, Max }

    public class AxisOpsExpr<U> : Expr<U>
    {
        public AxisOpsExpr(AxisOpsType axisOps, Expr<U> expr, int axis, bool keepdims)
        {
            foreach (var e in expr.Parameters)
                if (!Parameters.ContainsKey(e.Key))
                    Parameters[e.Key] = e.Value;

            Name = $"{axisOps} axis:{axis}";
            this.axisOps = axisOps;
            this.expr = expr;
            this.axis = axis;
            this.keepdims = keepdims;
            this.expr.IsRoot = false;
        }

        readonly AxisOpsType axisOps;
        readonly Expr<U> expr;
        readonly int axis;
        readonly bool keepdims;

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
            if (axisOps == AxisOpsType.Sum)
                ND.ApplyAxisOps(expr, this, axis, keepdims, NDarray<U>.OpsT.Add, NDarray<U>.OpsT.Zero);
            else if (axisOps == AxisOpsType.Prod)
                ND.ApplyAxisOps(expr, this, axis, keepdims, NDarray<U>.OpsT.Mul, NDarray<U>.OpsT.One);
            else if (axisOps == AxisOpsType.Min)
                ND.ApplyAxisOps(expr, this, axis, keepdims, NDarray<U>.OpsT.Min, NDarray<U>.OpsT.Maxvalue);
            else if (axisOps == AxisOpsType.Max)
                ND.ApplyAxisOps(expr, this, axis, keepdims, NDarray<U>.OpsT.Max, NDarray<U>.OpsT.Minvalue);
            else if (axisOps == AxisOpsType.Mean)
                ND.ApplyAxisOps(expr, this, axis, keepdims, NDarray<U>.OpsT.Add, NDarray<U>.OpsT.Zero, true);
        }
    }
}
