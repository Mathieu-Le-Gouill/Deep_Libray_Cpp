#pragma once
#include "layers/layer.h"
#include "layers/conv.h"
#include "layers/relu.h"
#include "layers/dense.h"
#include "layers/flatten.h"
#include "layers/norm.h"
#include "layers/sigmoid.h"
#include "layers/softmax.h"

#include <utility>
#include <tuple>
#include <type_traits>

template <::std::size_t ... Is>
constexpr auto indexSequenceReverse(::std::index_sequence<Is...> const&)
-> decltype(::std::index_sequence<sizeof...(Is) - 1U - Is...>{});

template <::std::size_t N>
using makeIndexSequenceReverse
= decltype(indexSequenceReverse(::std::make_index_sequence<N>{}));

template <typename... Layers>
class Pipeline {
public:
    template <::std::size_t... Dims>
    auto forward(Tensor<Dims...> input) {
        return processLayers(input, std::index_sequence_for<Layers...>(), [](auto&& layer, auto&& data) {
            return layer.Forward(std::forward<decltype(data)>(data));
        });
    }

    template <::std::size_t... Dims>
    auto forward(Tensor<Dims...>&& input) {
        return processLayers(input, std::index_sequence_for<Layers...>(), [](auto&& layer, auto&& data) {
            return layer.Forward(std::forward<decltype(data)>(data));
        });
    }

    template <::std::size_t... Dims>
    auto backward(Tensor<Dims...> input) {
        return processLayers(input, makeIndexSequenceReverse<sizeof...(Layers)>(), [](auto&& layer, auto&& data) {
            return layer.Backward(std::forward<decltype(data)>(data));
        });
    }

    template <::std::size_t... Dims>
    auto backward(Tensor<Dims...>&& input) {
        return processLayers(input, makeIndexSequenceReverse<sizeof...(Layers)>(), [](auto&& layer, auto&& data) {
            return layer.Backward(std::forward<decltype(data)>(data));
        });
    }

    void update() {
        // Update all layers
        std::apply([](auto&&... layer) {((layer.Update()), ...); }, layers);
    }

private:
    std::tuple<Layers...> layers;

    template <::std::size_t... Dims, size_t Index, typename Func, size_t... Rest>
    auto processLayers(Tensor<Dims...>& input, std::index_sequence<Index, Rest...>, Func&& func) {
        // Forward the input through the current layer
        auto output = func(std::get<Index>(layers), input);

        // Continue the forward pass with the rest of the layers
        return processLayers(output, std::index_sequence<Rest...>(), func);
    }

    template <::std::size_t... Dims, typename Func>
    auto processLayers(Tensor<Dims...>& input, std::index_sequence<>, Func&& func) {
        // When there are no more layers, return the final output
        return input;
    }
};
