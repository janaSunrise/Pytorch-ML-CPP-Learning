#include <torch/torch.h>

#include <iostream>

// Hyper parameters
const int64_t sequence_length = 28;

const int64_t input_size = 28;

const int64_t hidden_size = 128;

const int64_t num_layers = 2;

const int64_t num_classes = 10;

const int64_t batch_size = 100;

const size_t num_epochs = 10;

const double learning_rate = 0.01;

const std::string data_path = "../../../data/mnist/";

// Neural Net
struct Net : torch::nn::Module {
    Net() :
        lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
        fc(hidden_size, num_classes)
    {
        register_module("lstm", lstm);
        register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = std::get<0>(lstm->forward(x)).index({torch::indexing::Slice(), -1});
        return fc->forward(out);
    }

    torch::nn::LSTM lstm;
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

    // Create model
    Net model;
    model.to(device);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training..." << std::endl;

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            // Transfer images and target labels to device
            auto data = batch.data.view({-1, sequence_length, input_size}).to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model.forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / train_dataset_size;
        auto accuracy = static_cast<float>(num_correct) / train_dataset_size;

        std::cout << "Epoch ["
                  << (epoch + 1) << "/" << num_epochs
                  << "], Train - Loss: "
                  << sample_mean_loss << " | Accuracy: " << accuracy << std::endl;
    }

    std::cout << "Training finished!\n" << std::endl;
    std::cout << "Testing..." << std::endl;

    // Test the model
    model.eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.view({-1, sequence_length, input_size}).to(device);
        auto target = batch.target.to(device);

        auto output = model.forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);
        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!" << std::endl;

    auto test_accuracy = static_cast<double>(num_correct) / test_dataset_size;
    auto test_sample_mean_loss = running_loss / test_dataset_size;

    std::cout << "Testing | Loss: " << test_sample_mean_loss << " | Accuracy: " << test_accuracy << std::endl;
}
