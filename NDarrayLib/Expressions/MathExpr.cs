using System;
using System.Text;

namespace NDarrayLib.Expressions
{
    public enum FuncName
    {
        Identity,
        Abs,
        Exp,
        Log,
        Neg,
        Sq,
        Sqrt,
        Sigmoid,
        Tanh,
        DSigmoid,
        DTanh,
        AddLeft,
        AddRight,
        SubLeft,
        SubRight,
        MulLeft,
        MulRight,
        DivLeft,
        DivRight
    }

    public class MathExpr<U> : Expr<U>
    {
        public MathExpr(FuncName name, Expr<U> expr) : this(name, 0, expr) { }

        public MathExpr(FuncName name, double a, Expr<U> expr) : this(name, NDarray<U>.OpsT.Cast(a), expr) { }

        public MathExpr(FuncName name, U a, Expr<U> expr)
        {
            foreach (var e in expr.Parameters)
                if (!Parameters.ContainsKey(e.Key))
                    Parameters[e.Key] = e.Value;

            var infos = MathFunc.Infos(name, a);
            if (expr is MathExpr<U> mf)
            {
                (Name, Fnc) = MathFunc.Compose((mf.Name, mf.Fnc), infos);
                this.expr = mf.expr;
            }
            else
            {
                (Name, Fnc) = infos;
                this.expr = expr;
            }

            this.expr.IsRoot = false;
        }

        readonly Expr<U> expr;
        Func<U, U> Fnc;

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

    public static class MathFunc
    {
        public static (string, Func<U, U>) Compose<U>((string, Func<U, U>) first, (string, Func<U, U>) last)
        {
            var s = last.Item1.Replace("#", first.Item1);
            Func<U, U> f = x => last.Item2(first.Item2(x));
            return (s, f);
        }

        public static (string, Func<U, U>) Infos<U>(FuncName name, U a)
        {
            if (name == FuncName.Abs)
                return ("|#|", x => NDarray<U>.OpsT.Abs(x));
            if (name == FuncName.Exp)
                return ("Exp[#]", x => NDarray<U>.OpsT.Exp(x));
            if (name == FuncName.Log)
                return ("Log[#]", x => NDarray<U>.OpsT.Log(x));
            if (name == FuncName.Neg)
                return ("-[#]", x => NDarray<U>.OpsT.Neg(x));
            if (name == FuncName.Sq)
                return ("[#]^2", x => NDarray<U>.OpsT.Sq(x));
            if (name == FuncName.Sqrt)
                return ("Sqrt[#]", x => NDarray<U>.OpsT.Sqrt(x));
            if (name == FuncName.Sigmoid)
                return ("Sigmoid[#]", x => NDarray<U>.OpsT.Sigmoid(x));
            if (name == FuncName.DSigmoid)
                return ("DSigmoid[#]", x => NDarray<U>.OpsT.DSigmoid(x));
            if (name == FuncName.Tanh)
                return ("Tanh[#]", x => NDarray<U>.OpsT.Tanh(x));
            if (name == FuncName.DTanh)
                return ("DTanh[#]", x => NDarray<U>.OpsT.DTanh(x));

            if (name == FuncName.AddLeft)
                return ($"{a} + [#]", x => NDarray<U>.OpsT.Add(a, x));
            if (name == FuncName.AddRight)
                return ($"[#] + {a}", x => NDarray<U>.OpsT.Add(x, a));
            if (name == FuncName.SubLeft)
                return ($"{a} - [#]", x => NDarray<U>.OpsT.Sub(a, x));
            if (name == FuncName.SubRight)
                return ($"[#] - {a}", x => NDarray<U>.OpsT.Sub(x, a));
            if (name == FuncName.MulLeft)
                return ($"{a} * [#]", x => NDarray<U>.OpsT.Mul(a, x));
            if (name == FuncName.MulRight)
                return ($"[#] * {a}", x => NDarray<U>.OpsT.Mul(x, a));
            if (name == FuncName.DivLeft)
                return ($"{a} / [#]", x => NDarray<U>.OpsT.Div(a, x));
            if (name == FuncName.DivRight)
                return ($"[#] / {a}", x => NDarray<U>.OpsT.Div(x, a));

            return ("#", x => x);
        }
    }
}
