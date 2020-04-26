#ifndef NN_RELU_LAYER_HPP
#define NN_RELU_LAYER_HPP

#include <nn/Layer/Layer.hpp>

#include <algorithm>

namespace nn
{
class ReLULayer : public Layer
{
 public:
    ReLULayer(std::size_t outSize)
        : Layer(outSize, outSize)
    {
    }

    ReLULayer(const ReLULayer&) = delete;
    ReLULayer(ReLULayer&&) = delete;
    ReLULayer& operator=(const ReLULayer&) = delete;
    ReLULayer& operator=(ReLULayer&&) = delete;

 private:
    void forward_impl(const Layer::Tensor& input, Layer::Tensor& output) override
    {
        const std::size_t inSize = InputSize();

        #pragma omp parallel for
        for (long long i = 0; i < inSize; ++i)
        {
            output[i] = std::max(input[i], 0.f);
        }
    }
};
}  // namespace nn

#endif  // NN_RELU_LAYER_HPP
