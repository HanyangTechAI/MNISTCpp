#include <iostream>

#include <nn/Layer/ReLULayer.hpp>
#include <nn/Model/Sequential.hpp>

int main()
{
    constexpr std::size_t N = 100000000;

    nn::Sequential net;

    net.PushLayer<nn::ReLULayer>(N);
    net.PushLayer<nn::ReLULayer>(N);
    net.PushLayer<nn::ReLULayer>(N);
    net.PushLayer<nn::ReLULayer>(N);
    net.PushLayer<nn::ReLULayer>(N);
    net.PushLayer<nn::ReLULayer>(N);

    nn::Layer::Tensor input(N);

    net.Forward(input);
}
