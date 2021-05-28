#include <torch/torch.h>

#include <iostream>

// Hyperparameters
const int64_t sequence_length = 28;

const int64_t input_size = 28;

const int64_t hidden_size = 128;

const int64_t num_layers = 2;

const int64_t num_classes = 10;

const int64_t batch_size = 100;

const size_t num_epochs = 10;

const double learning_rate = 0.01;

const std::string data_path = "../../../data/mnist/";

// Model
struct Net : torch::nn::Module {
    Net() :
        lstm(
            torch::nn::LSTMOptions(input_size, hidden_size)
                       .num_layers(num_layers)
                       .batch_first(true)
                       .bidirectional(true)
        ),
        fc(hidden_size * 2, num_classes)
    {
        register_module("lstm", lstm);
        register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = std::get<0>(lstm->forward(x)).index({torch::indexing::Slice(), -1});

        auto out_directions = out.chunk(2, 2);
        auto out_1 = out_directions[0].index({Slice(), -1});
        auto out_2 = out_directions[1].index({Slice(), 0});
        auto out_final = torch::cat({out_1, out_2}, 1);

        return fc->forward(out_final);
    }

    torch::nn::LSTM lstm;
    torch::nn::Linear fc;
};

// Main function
auto main() -> int {
    std::cout << "Welcome to Bidirectional RNNs." << std::endl;
}
