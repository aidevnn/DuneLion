using System;
using System.Text;

namespace NDarrayLib.Expressions
{
    public enum VariableType { Scalar, NDarray }
    public abstract class Variable
    {
        public Guid Guid { get; protected set; }
        public string Name { get; protected set; }
        public Type Dtype { get; protected set; }

        public abstract VariableType Vtype { get; }
        public abstract void SetContent(object o);
        public abstract object GetContent { get; }

        class VariableScalar<U> : Variable
        {
            public NDarray<U> Scalar { get; private set; }
            public override VariableType Vtype => VariableType.Scalar;
            public override object GetContent => Scalar;

            public VariableScalar()
            {
                Guid = Guid.NewGuid();
                Name = Guid.ToString();
                Dtype = typeof(U);
                Scalar = new NDarray<U>(shape: new int[] { 1 });
            }

            public VariableScalar(U scalar) : this() { Scalar.Data[0] = scalar; }
            public VariableScalar(string name) : this() { Name = name; }
            public VariableScalar(string name, U scalar) : this(scalar) { Name = name; }
            public override void SetContent(object o) => Scalar.Data[0] = NDarray<U>.OpsT.Cast(o);

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"Name: {Name,-8} {Vtype}<{Dtype.Name}> Guid:{Guid} HashCode:{GetHashCode()}");
                sb.Append($"{Scalar.Data[0]}");
                return sb.ToString();
            }
        }

        class VariableNDarray<U> : Variable
        {
            public NDarray<U> NDarray { get; private set; }
            public override VariableType Vtype => VariableType.NDarray;
            public override object GetContent => NDarray;

            public VariableNDarray()
            {
                Guid = Guid.NewGuid();
                Name = Guid.ToString();
                Dtype = typeof(U);
                NDarray = null;
            }

            public VariableNDarray(NDarray<U> nDarray) : this() { NDarray = nDarray; }
            public VariableNDarray(string name) : this() { Name = name; }
            public VariableNDarray(string name, NDarray<U> nDarray) : this(nDarray) { Name = name; }
            public override void SetContent(object o)
            {
                if (!(o is NDarray<U> o0))
                    throw new ArgumentException("NDarray bad type");

                if (NDarray == null)
                    NDarray = o0.Copy;
                else if (NDarray.Shape.Length == o0.Shape.Length && Utils.SameShape(NDarray.Shape, o0.Shape))
                {
                    NDarray.CopyData(o0);
                }
                else
                {
                    NDarray = o0.Copy;
                    //throw new ArgumentException("NDarray must have the same data type and the same shape");
                }
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"Name: {Name,-8} {Vtype}<{Dtype.Name}> Guid:{Guid} HashCode:{GetHashCode()}");
                sb.Append($"{NDarray}");
                return sb.ToString();
            }
        }

        public static NullaryExpr<U> CreateScalar<U>() => new NullaryExpr<U>(new VariableScalar<U>());
        public static NullaryExpr<U> CreateScalar<U>(string name) => new NullaryExpr<U>(new VariableScalar<U>(name));
        public static NullaryExpr<U> CreateScalar<U>(U v) => new NullaryExpr<U>(new VariableScalar<U>(v));
        public static NullaryExpr<U> CreateScalar<U>(string name, U v) => new NullaryExpr<U>(new VariableScalar<U>(name, v));

        public static NullaryExpr<U> CreateNDarray<U>() => new NullaryExpr<U>(new VariableNDarray<U>());
        public static NullaryExpr<U> CreateNDarray<U>(string name) => new NullaryExpr<U>(new VariableNDarray<U>(name));
        public static NullaryExpr<U> CreateNDarray<U>(NDarray<U> nDarray) => new NullaryExpr<U>(new VariableNDarray<U>(nDarray));
        public static NullaryExpr<U> CreateNDarray<U>(string name, NDarray<U> nDarray) => new NullaryExpr<U>(new VariableNDarray<U>(name, nDarray));
    }

}
