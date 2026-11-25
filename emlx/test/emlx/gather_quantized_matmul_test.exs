defmodule EMLX.GatherQuantizedMatmulTest do
  @moduledoc """
  Tests for `EMLX.Quantization.gather_quantized_matmul/4` — the `Nx.Tensor`
  level entry point for `mx::gather_qmm`, usable inside compiled numerical
  definitions.
  """
  use EMLX.Case, async: true

  @moduletag :metal

  defp stacked_experts(count, out, inp) do
    Nx.iota({count, out, inp}, type: :f32)
    |> Nx.divide(count * out * inp)
    |> Nx.subtract(0.5)
    |> EMLX.quantize([])
  end

  # Each row multiplied by its own expert, spelled out with dense matmuls.
  defp reference(x, qw, indices) do
    dense = EMLX.Quantization.dequantize(qw)

    indices
    |> Nx.to_flat_list()
    |> Enum.with_index()
    |> Enum.map(fn {expert, row} -> Nx.dot(x[row], [2], dense[expert], [1]) end)
    |> Nx.stack()
  end

  describe "Nx.dot of experts taken from a quantized tensor" do
    defp take_dot(x, w, indices), do: Nx.dot(x, [1], [0], Nx.take(w, indices), [2], [0])

    test "multiplies each row by its expert, as with the dequantized experts" do
      qw = stacked_experts(4, 32, 128)
      x = Nx.iota({5, 128}, type: :f32) |> Nx.divide(128)
      indices = Nx.tensor([3, 0, 3, 1, 2])

      got = Nx.Defn.jit(&take_dot/3, compiler: EMLX).(x, qw, indices)
      expected = take_dot(x, EMLX.Quantization.dequantize(qw), indices)

      assert Nx.shape(got) == {5, 32}
      assert_all_close(got, expected, atol: 1.0e-3)
    end

    test "matches gather_quantized_matmul/4" do
      qw = stacked_experts(4, 32, 128)
      x = Nx.iota({4, 128}, type: :f32) |> Nx.divide(128)
      indices = Nx.tensor([2, 2, 0, 1])

      got = Nx.Defn.jit(&take_dot/3, compiler: EMLX).(x, qw, indices)

      expected =
        x
        |> Nx.reshape({4, 1, 1, 128})
        |> EMLX.Quantization.gather_quantized_matmul(
          qw,
          Nx.reshape(Nx.as_type(indices, :u32), {4, 1})
        )
        |> Nx.reshape({4, 32})

      assert_all_close(got, expected)
    end
  end

  describe "gather_quantized_matmul/4" do
    test "matches a dense per-expert reference" do
      qw = stacked_experts(4, 32, 128)
      x = Nx.iota({4, 1, 1, 128}, type: :f32) |> Nx.divide(128)
      indices = Nx.tensor([[0], [1], [2], [3]], type: :u32)

      got = EMLX.Quantization.gather_quantized_matmul(x, qw, indices)

      assert Nx.shape(got) == {4, 1, 1, 32}
      assert_all_close(got, reference(x, qw, indices), atol: 1.0e-3)
    end

    test "the same expert may be named by several rows" do
      qw = stacked_experts(4, 16, 64)
      x = Nx.iota({4, 1, 1, 64}, type: :f32) |> Nx.divide(64)
      indices = Nx.tensor([[2], [2], [0], [2]], type: :u32)

      got = EMLX.Quantization.gather_quantized_matmul(x, qw, indices)

      assert_all_close(got, reference(x, qw, indices), atol: 1.0e-3)
    end

    test "runs inside a compiled numerical definition" do
      qw = stacked_experts(4, 32, 128)
      x = Nx.iota({4, 1, 1, 128}, type: :f32) |> Nx.divide(128)
      indices = Nx.tensor([[0], [1], [2], [3]], type: :u32)

      fun =
        Nx.Defn.compile(
          &EMLX.Quantization.gather_quantized_matmul/3,
          [x, qw, indices],
          compiler: EMLX
        )

      # On :gpu the compiled and the eager results can differ by 1 ulp.
      assert_all_close(
        fun.(x, qw, indices),
        EMLX.Quantization.gather_quantized_matmul(x, qw, indices)
      )
    end

    test "sorted_indices takes the same path when indices are ordered" do
      qw = stacked_experts(4, 16, 64)
      x = Nx.iota({4, 1, 1, 64}, type: :f32) |> Nx.divide(64)
      indices = Nx.tensor([[0], [1], [2], [3]], type: :u32)

      assert_all_close(
        EMLX.Quantization.gather_quantized_matmul(x, qw, indices, sorted_indices: true),
        EMLX.Quantization.gather_quantized_matmul(x, qw, indices),
        atol: 1.0e-3
      )
    end

    test "rejects a dense second argument" do
      x = Nx.iota({4, 1, 1, 64}, type: :f32)
      dense = Nx.iota({4, 16, 64}, type: :f32)
      indices = Nx.tensor([[0], [1], [2], [3]], type: :u32)

      assert_raise ArgumentError, ~r/must be a quantized tensor/, fn ->
        EMLX.Quantization.gather_quantized_matmul(x, dense, indices)
      end
    end
  end
end
