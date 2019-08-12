using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NDarrayLib.Expressions
{
    public abstract class Expr<U>
    {
        public string Name { get; set; }
        public bool IsRoot { get; set; } = true;
        public Dictionary<string, Variable> Parameters = new Dictionary<string, Variable>();

        public Variable Result = null;

        public abstract void Eval();
        public abstract string Display(string indent = "");

        public void SetParameters(params (string, object)[] args)
        {
            foreach (var a in args)
                Parameters[a.Item1].SetContent(a.Item2);
        }

        public NDarray<U> Evaluate(params (string, object)[] args)
        {
            SetParameters(args);
            Eval();
            return Result.GetContent as NDarray<U>;
        }

        public void SetContent(U o)
        {
            if (Result == null)
                Result = Variable.CreateScalar(o).Result;
            Result.SetContent(o);

        }

        public void SetContent(NDarray<U> o)
        {
            if (Result == null)
                Result = Variable.CreateNDarray(o).Result;
            Result.SetContent(o);
        }

        public NDarray<U> GetContent => Result.GetContent as NDarray<U>;

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(Display());
            sb.AppendLine();
            sb.AppendLine($"{Result}");

            return sb.ToString();
        }

        public Expr<V> Cast<V>() => ND.DoCast<U, V>(this);

        public Expr<U> Reshape(params int[] shape) => ND.Reshape(this, shape);
        public Expr<U> Transpose(params int[] table) => ND.Transpose(this, table);
        public Expr<U> T => Transpose();

        public Expr<U> Sum(int axis = -1, bool keepdims = false) => ND.SumAxis(this, axis, keepdims);
        public Expr<U> Prod(int axis = -1, bool keepdims = false) => ND.ProdAxis(this, axis, keepdims);
        public Expr<U> Mean(int axis = -1, bool keepdims = false) => ND.MeanAxis(this, axis, keepdims);

        public Expr<double> MeanAll() => Cast<double>().Mean();

        public static implicit operator Expr<U>(Variable variable) => new NullaryExpr<U>(variable);
        public static implicit operator NDarray<U>(Expr<U> expr) => expr.Evaluate();

        public static Expr<U> operator +(double a, Expr<U> expr) => new MathExpr<U>(FuncName.AddLeft, a, expr);
        public static Expr<U> operator +(Expr<U> expr, double a) => new MathExpr<U>(FuncName.AddRight, a, expr);
        public static Expr<U> operator +(Expr<U> left, Expr<U> right) => ND.Add(left, right);

        public static Expr<U> operator -(Expr<U> expr) => new MathExpr<U>(FuncName.Neg, expr);
        public static Expr<U> operator -(double a, Expr<U> expr) => new MathExpr<U>(FuncName.SubLeft, a, expr);
        public static Expr<U> operator -(Expr<U> expr, double a) => new MathExpr<U>(FuncName.SubRight, a, expr);
        public static Expr<U> operator -(Expr<U> left, Expr<U> right) => ND.Sub(left, right);

        public static Expr<U> operator *(double a, Expr<U> expr) => new MathExpr<U>(FuncName.MulLeft, a, expr);
        public static Expr<U> operator *(Expr<U> expr, double a) => new MathExpr<U>(FuncName.MulRight, a, expr);
        public static Expr<U> operator *(Expr<U> left, Expr<U> right) => ND.Mul(left, right);

        public static Expr<U> operator /(double a, Expr<U> expr) => new MathExpr<U>(FuncName.DivLeft, a, expr);
        public static Expr<U> operator /(Expr<U> expr, double a) => new MathExpr<U>(FuncName.DivRight, a, expr);
        public static Expr<U> operator /(Expr<U> left, Expr<U> right) => ND.Div(left, right);

    }
}
