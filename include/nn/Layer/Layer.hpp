#ifndef NN_LAYER_HPP
#define NN_LAYER_HPP

#include <cassert>
#include <cctype>
#include <vector>

namespace nn
{
template <typename DT>
class Layer
{
 public:
    Layer(std::size_t inSize, std::size_t outSize)
        : inSize_(inSize), outSize_(outSize)
    {
        output_.resize(outSize);
    }
    virtual ~Layer() = default;

    Layer(const Layer&) = delete;
    Layer(Layer&&) = delete;
    Layer& operator=(const Layer&) = delete;
    Layer& operator=(Layer&&) = delete;

    std::size_t InputSize() const
    {
        return inSize_;
    }

    std::size_t OutputSize() const
    {
        return outSize_;
    }

    void Forward(const std::vector<DT>& input)
    {
        assert(input.size() == inSize_);

        forward_impl(input, output_);
    }

    const std::vector<DT>& Output() const
    {
        return output_;
    }

 private:
    virtual void forward_impl(const std::vector<DT>& input, std::vector<DT>& output) = 0;

 private:
    std::size_t inSize_;
    std::size_t outSize_;

    std::vector<DT> output_;
};
}  // namespace nn

#endif  // NN_LAYER_HPP
