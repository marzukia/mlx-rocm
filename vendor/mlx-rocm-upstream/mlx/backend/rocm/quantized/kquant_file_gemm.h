// File-layout k-quant ops — fast-path primitive classes.
//
// See mlx/kquant_file_gemm.h for the op semantics. The three classes
// (GEMM / gather / standalone dequant) are Custom (fallback-throws) like
// KQuantDequantize: GPU-only (ROCm build), inference-time ops.
#pragma once

#include <tuple>

#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

class KQuantFileGemm : public Custom {
 public:
  explicit KQuantFileGemm(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int ktype,
      int mode,
      int axis,
      int group,
      int keep_ends,
      int pairs)
      : Custom(stream, std::move(fallback)),
        ktype_(ktype),
        mode_(mode),
        axis_(axis),
        group_(group),
        keep_ends_(keep_ends),
        pairs_(pairs) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("KQuantFileGemm only runs on GPU.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(KQuantFileGemm);
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(
        nullptr, ktype_, mode_, axis_, group_, keep_ends_, pairs_);
  }

 private:
  int ktype_;
  int mode_;
  int axis_;
  int group_;
  int keep_ends_;
  int pairs_;
};

class KQuantFileGather : public Custom {
 public:
  explicit KQuantFileGather(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int ktype,
      int mode,
      int axis,
      int group,
      int keep_ends,
      int pairs,
      Dtype out_dtype)
      : Custom(stream, std::move(fallback)),
        ktype_(ktype),
        mode_(mode),
        axis_(axis),
        group_(group),
        keep_ends_(keep_ends),
        pairs_(pairs),
        out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("KQuantFileGather only runs on GPU.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(KQuantFileGather);
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(
        nullptr, ktype_, mode_, axis_, group_, keep_ends_, pairs_, out_dtype_);
  }

 private:
  int ktype_;
  int mode_;
  int axis_;
  int group_;
  int keep_ends_;
  int pairs_;
  Dtype out_dtype_;
};

class KQuantFileDequant : public Custom {
 public:
  explicit KQuantFileDequant(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int ktype,
      int mode,
      int axis,
      int group,
      int keep_ends,
      int pairs)
      : Custom(stream, std::move(fallback)),
        ktype_(ktype),
        mode_(mode),
        axis_(axis),
        group_(group),
        keep_ends_(keep_ends),
        pairs_(pairs) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("KQuantFileDequant only runs on GPU.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(KQuantFileDequant);
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(
        nullptr, ktype_, mode_, axis_, group_, keep_ends_, pairs_);
  }

 private:
  int ktype_;
  int mode_;
  int axis_;
  int group_;
  int keep_ends_;
  int pairs_;
};

}  // namespace mlx::core::fast
