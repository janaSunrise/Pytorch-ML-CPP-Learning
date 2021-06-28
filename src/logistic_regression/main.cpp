#include <iostream>

// Hyperparameters
const int64_t input_size = 784;

const int64_t num_classes = 10;

const int64_t batch_size = 100;

const size_t num_epochs = 5;

const double learning_rate = 0.001;

const std::string data_path = "../../../data/mnist/";

int main() {
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

    // Datasets
    auto train_dataset = torch::data::datasets::MNIST(data_path)
                         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                         .map(torch::data::transforms::Stack<>());

    const size_t train_dataset_size = train_dataset.size().value();

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), batch_size
    );

    auto test_dataset = torch::data::datasets::MNIST(data_path, torch::data::datasets::MNIST::Mode::kTest)
                        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                        .map(torch::data::transforms::Stack<>());

    const size_t test_dataset_size = test_dataset.size().value();

    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), batch_size);
}
