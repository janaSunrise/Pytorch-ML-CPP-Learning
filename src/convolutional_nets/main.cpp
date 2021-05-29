#include <iostream>

// Hyper parameters
const int64_t num_classes = 10;

const int64_t batch_size = 100;

const size_t num_epochs = 5;

const double learning_rate = 0.001;

const std::string MNIST_data_path = "../../../data/mnist/";

// Model
struct Net : torch::nn::Module {
    Net() :
        fc(7 * 7 * 32, num_classes)
    {
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = layer1->forward(x);
        x = layer2->forward(x);

        x = x.view({-1, 7 * 7 * 32});

        return fc->forward(x);
    }

    torch::nn::Sequential layer1{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)),
        torch::nn::BatchNorm2d(16),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer2{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Linear fc;
};

auto main() -> int {
    // Set the seed
    torch::manual_seed(1);

    // Get the type of device
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    torch::Device device(device_type);
}
