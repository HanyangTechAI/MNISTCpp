#ifndef NN_SEQUENTIAL_HPP
#define NN_SEQUENTIAL_HPP

#include <nn/Layer/Layer.hpp>

#include <cassert>
#include <memory>
#include <vector>

namespace nn
{
class Sequential final
{
 public:
    Sequential() = default;

    Sequential(const Sequential&) = delete;
    Sequential(Sequential&&) = delete;
    Sequential& operator=(const Sequential&) = delete;
    Sequential& operator=(Sequential&&) = delete;

    template <class LayerT, typename... Args>
    void PushLayer(Args&&... args)
    {
        layers_.emplace_back(
            std::make_unique<LayerT>(std::forward<Args>(args)...));
    }

    const Layer::Tensor& Forward(const Layer::Tensor& input)
    {
        assert(layers_.empty() == false);

        const std::size_t totalLevel = layers_.size();

        layers_.front()->Forward(input);

        for (std::size_t level = 1; level < totalLevel; ++level)
        {
            layers_[level]->Forward(layers_[level - 1]->Output());
        }

        return layers_.back()->Output();
    }

 private:
    std::vector<std::unique_ptr<Layer>> layers_;
};
}  // namespace nn

#endif  // NN_SEQUENTIAL_HPP
