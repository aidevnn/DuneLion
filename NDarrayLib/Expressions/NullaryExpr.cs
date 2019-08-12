using System;
namespace NDarrayLib.Expressions
{
    public class NullaryExpr<U> : Expr<U>
    {
        public NullaryExpr(Variable variable)
        {
            if (variable.Dtype != typeof(U))
                throw new ArgumentException("Variable Type not match Expression Type");

            Result = variable;
            Name = "Nullary";
            Parameters[variable.Name] = variable;
        }

        //public void SetContent(U o) => Result.SetContent(o);
        //public void SetContent(NDarray<U> o) => Result.SetContent(o);
        //public NDarray<U> GetContent => Result.GetContent as NDarray<U>;

        public override string Display(string indent)
        {
            return $"{indent}Variable {Result.Name} {Result.Vtype}<{Result.Dtype.Name}>\n";
        }

        public override void Eval() { }
    }
}
