using System;
using System.Text;

namespace NDarrayLib.Expressions
{
    public enum ArgIndex { Min, Max }

    public class ArgMinMaxExpr<U> : Expr<int>
    {
        public ArgMinMaxExpr(ArgIndex argIndex, int axis, Expr<U> expr)
        {
            foreach (var e in expr.Parameters)
                if (!Parameters.ContainsKey(e.Key))
                    Parameters[e.Key] = e.Value;

            Name = $"Arg{argIndex} axis:{axis}";
            this.argIndex = argIndex;
            this.axis = axis;
            this.expr = expr;
            this.expr.IsRoot = false;
        }

        readonly ArgIndex argIndex;
        readonly int axis;
        readonly Expr<U> expr;

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
            if (argIndex == ArgIndex.Min)
                ND.ApplyArgMinMax(expr, this, axis, NDarray<U>.OpsT.Min, NDarray<U>.OpsT.Maxvalue);
            else
                ND.ApplyArgMinMax(expr, this, axis, NDarray<U>.OpsT.Max, NDarray<U>.OpsT.Minvalue);
        }
    }
}
