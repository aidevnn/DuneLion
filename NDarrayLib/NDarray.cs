using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NDarrayLib
{
    public class NDarray<U>
    {
        public static Operations<U> OpsT;

        static NDarray()
        {
            if (typeof(U) == typeof(int))
                OpsT = new OpsInt() as Operations<U>;
            else if (typeof(U) == typeof(float))
                OpsT = new OpsFloat() as Operations<U>;
            else if (typeof(U) == typeof(double))
                OpsT = new OpsDouble() as Operations<U>;
            else
                throw new ArgumentException($"{typeof(U).Name} is not supported. Only int, float or double");
        }

        public int[] Shape { get; protected set; }
        public int[] Strides { get; protected set; }
        public int[] Indices { get; protected set; }
        public int Count { get; protected set; }

        public U[] Data;

        internal NDarray(int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            Shape = shape.ToArray();
            Strides = Utils.Shape2Strides(Shape);
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            Data = new U[Count];
        }

        public NDarray(int[] shape, int[] strides)
        {
            if (shape.Length == 0)
            {
                shape = new int[] { 1 };
                strides = new int[] { 1 };
            }

            Shape = shape.ToArray();
            Strides = strides.ToArray();
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            Data = new U[Count];
        }

        public NDarray(U v0, int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            Shape = shape.ToArray();
            Strides = Utils.Shape2Strides(Shape);
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            Data = new U[Count];
            for (int idx = 0; idx < Count; ++idx)
                Data[idx] = v0;
        }

        public NDarray(U[] data, int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            Shape = shape.ToArray();
            Strides = Utils.Shape2Strides(Shape);
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            if (data.Length != Count)
                throw new ArgumentException();

            Data = data.ToArray();
        }

        public NDarray(NDarray<U> nDarray)
        {
            Shape = nDarray.Shape.ToArray();
            Strides = nDarray.Strides.ToArray();
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            Data = nDarray.Data.ToArray();
        }

        public NDarray<U> Copy => new NDarray<U>(this);

        public void CopyData(NDarray<U> nDarray) => nDarray.Data.CopyTo(Data, 0);

        public NDarray<V> Cast<V>()
        {
            var data = Data.Select(NDarray<V>.OpsT.Cast).ToArray();
            return new NDarray<V>(data: data, shape: Shape);
        }

        public U[] DataAtIdx(int k)
        {
            int nb = Utils.ArrMul(Shape, 1);
            int start = k * Strides[0];
            return Enumerable.Range(start, nb).Select(i => Data[i]).ToArray();
        }

        public override string ToString()
        {
            var nargs = new int[Shape.Length];
            var strides = Utils.Shape2Strides(Shape);

            string result = "";
            var last = strides.Length == 1 ? Count : strides[strides.Length - 2];
            string before, after;

            List<U> listValues = Data.ToList();

            if (Utils.IsDebugLvl1)
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"Class:{GetType().Name,-20}");
                if (Shape.Length > 1 || Shape[0] != 1)
                {
                    sb.AppendLine($"Shape:({Shape.Glue()}) Version:{GetHashCode(),10}");
                    sb.AppendLine($"Strides:({Strides.Glue()})");
                }

                string dbg = $" : np.array([{listValues.Glue(",")}], dtype={OpsT.dtype}).reshape({Shape.Glue(",")})";
                var nd = $"NDArray<{typeof(U).Name}>";
                sb.AppendLine($"{nd,-20} {Shape.Glue("x")}{dbg}");
                result += sb.ToString();
            }

            var ml0 = listValues.Select(v => $"{v}").Max(v => v.Length);
            var ml1 = listValues.Select(v => $"{v:F8}").Max(v => v.Length);
            string fmt = $"{{0,{ml0 + 2}}}";
            if (ml0 > ml1 + 3)
                fmt = $"{{0,{ml1 + 2}:F8}}";

            for (int idx = 0; idx < Count; ++idx)
            {
                after = before = "";

                if (idx % last == 0 || idx % last == last - 1)
                {
                    before = idx != 0 ? " " : "[";
                    after = idx == Count - 1 ? "]" : "";
                    for (int l = strides.Length - 2; l >= 0; --l)
                    {
                        if (idx % strides[l] == 0) before += "[";
                        else before = " " + before;

                        if (idx % strides[l] == strides[l] - 1) after += "]";
                    }
                }

                result += idx % last == 0 ? before : "";
                var val = listValues[idx];
                result += string.Format(fmt, val);
                result += idx % last == last - 1 ? after + "\n" : "";
                result += after.Length > 1 && idx != Count - 1 ? "\n" : "";
            }

            if (Utils.IsDebugNo)
                result = result.Substring(0, result.Length - 1);

            return result;
        }
    }
}
