#ifndef NN_SOFTMAX_LAYER_HPP
#define NN_SOFTMAX_LAYER_HPP

#include <nn/Layer/Layer.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

#include <iostream>

namespace nn
{
class SoftmaxLayer : public Layer
{
 public:
    SoftmaxLayer(std::size_t outSize)
        : Layer(outSize, outSize)
    {
    }

    SoftmaxLayer(const SoftmaxLayer&) = delete;
    SoftmaxLayer(SoftmaxLayer&&) = delete;
    SoftmaxLayer& operator=(const SoftmaxLayer&) = delete;
    SoftmaxLayer& operator=(SoftmaxLayer&&) = delete;

 private:
    void forward_impl(const Layer::Tensor& input, Layer::Tensor& output) override
    {
        const std::size_t inSize = InputSize();

        float alpha = *std::max_element(begin(input), end(input));

        #pragma omp parallel for
        for (long long i = 0; i < inSize; ++i)
        {
            output[i] = std::exp(input[i] - alpha);
        }

        float sum = std::accumulate(begin(output), end(output), 0.f);
        #pragma omp parallel for
        for (long long i = 0; i < inSize; ++i)
        {
            output[i] /= sum;
        }
    }
};
}  // namespace nn

#endif  // NN_SOFTMAX_LAYER_HPP
