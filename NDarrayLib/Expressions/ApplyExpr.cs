using System;
using System.Text;

namespace NDarrayLib.Expressions
{
    public class ApplyExpr<U, V> : Expr<V>
    {
        public ApplyExpr(Expr<U> expr, Func<U, V> func)
        {
            foreach (var e in expr.Parameters)
                if (!Parameters.ContainsKey(e.Key))
                    Parameters[e.Key] = e.Value;

            this.expr = expr;
            Fnc = func;
            Name = "Apply";
            this.expr.IsRoot = false;
        }

        public ApplyExpr(string name, Expr<U> expr, Func<U, V> func) : this(expr, func)
        {
            Name = name;
        }

        readonly Expr<U> expr;
        readonly Func<U, V> Fnc;

        public override string Display(string indent = "")
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"{indent}# => {Name}");
            sb.Append(expr.Display(indent + "|___"));
            return sb.ToString();
        }

        public override void Eval()
        {
            expr.Eval();
            ND.ApplyFunc(expr, Fnc, this);
        }
    }
}
