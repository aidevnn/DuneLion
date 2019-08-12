using System;
using System.Text;

namespace NDarrayLib.Expressions
{
    public class ElementWiseExpr<U, V> : Expr<V>
    {
        public ElementWiseExpr(Expr<U> leftExpr, Func<U, U, V> func, Expr<U> rightExpr)
        {
            foreach (var e in leftExpr.Parameters)
                if (!Parameters.ContainsKey(e.Key))
                    Parameters[e.Key] = e.Value;

            foreach (var e in rightExpr.Parameters)
                if (!Parameters.ContainsKey(e.Key))
                    Parameters[e.Key] = e.Value;

            Name = func.Method.Name;
            this.leftExpr = leftExpr;
            this.rightExpr = rightExpr;
            Fnc = func;
            this.leftExpr.IsRoot = false;
            this.rightExpr.IsRoot = false;
        }

        readonly Expr<U> leftExpr, rightExpr;
        readonly Func<U, U, V> Fnc;

        public override string Display(string indent = "")
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"{indent}{Name}");
            sb.Append(leftExpr.Display(indent + "|___"));
            sb.Append(rightExpr.Display(indent + "|___"));
            return sb.ToString();
        }

        public override void Eval()
        {
            leftExpr.Eval();
            rightExpr.Eval();
            ND.ApplyElementWiseOp(leftExpr, rightExpr, Fnc, this);
        }
    }
}
