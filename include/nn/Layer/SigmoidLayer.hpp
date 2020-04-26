#ifndef NN_SIGMOID_LAYER_HPP
#define NN_SIGMOID_LAYER_HPP

#include <nn/Layer/Layer.hpp>

#include <cmath>

namespace nn
{
class SigmoidLayer : public Layer
{
 public:
    SigmoidLayer(std::size_t outSize)
        : Layer(outSize, outSize)
    {
    }

    SigmoidLayer(const SigmoidLayer&) = delete;
    SigmoidLayer(SigmoidLayer&&) = delete;
    SigmoidLayer& operator=(const SigmoidLayer&) = delete;
    SigmoidLayer& operator=(SigmoidLayer&&) = delete;

 private:
    void forward_impl(const Layer::Tensor& input, Layer::Tensor& output) override
    {
        const std::size_t inSize = InputSize();

        #pragma omp parallel for
        for (long long i = 0; i < inSize; ++i)
        {
            output[i] = 1.f / (1 + std::exp(-input[i]));
        }
    }
};
}  // namespace nn

#endif  // NN_SIGMOID_LAYER_HPP
