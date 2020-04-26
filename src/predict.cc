#include <iostream>

#include <nn/Layer/SigmoidLayer.hpp>
#include <nn/Model/Sequential.hpp>

int main()
{
    nn::Sequential net;

    net.PushLayer<nn::SigmoidLayer>(4);
    net.PushLayer<nn::SigmoidLayer>(4);

    nn::Layer::Tensor input{ 2, 3, 4, 5 };

    const auto& output = net.Forward(input);

    for (const auto v : output)
        std::cout << v << ' ';

    std::cout << std::endl;
}
