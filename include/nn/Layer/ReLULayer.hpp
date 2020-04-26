#ifndef NN_RELU_LAYER_HPP
#define NN_RELU_LAYER_HPP

#include <nn/Layer/Layer.hpp>

#include <algorithm>
#include <execution>

namespace nn
{
template <typename DT>
class ReLULayer : public Layer<DT>
{
 public:
    ReLULayer(std::size_t outSize)
        : Layer<DT>(outSize, outSize)
    {
    }

 private:
    void forward_impl(const std::vector<DT>& input, std::vector<DT>& output) override
    {
        std::transform(std::execution::par, begin(input), end(input), begin(output), [](DT value) {
            return std::max<DT>(value, static_cast<DT>(0));
        });
    }
};
}  // namespace nn

#endif  // NN_RELU_LAYER_HPP
