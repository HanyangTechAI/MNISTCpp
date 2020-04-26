#include <iostream>

#include <nn/Layer/ReLULayer.hpp>

int main()
{
    nn::ReLULayer<int> layer(3);

    layer.Forward({-1, 3, 5});

    const auto& output = layer.Output();

    for (const auto& v : output)
        std::cout << v << ' ';

    std::cout << std::endl;
}
