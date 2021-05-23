#include <torch/torch.h>

#include <iostream>

int main() {
    std::cout << "=========== Empty Tensors ============" << std::endl;

    auto empty_tensor = torch::empty({4, 4});
    std::cout << "4x4 empty tensor:\n" << empty_tensor << std::endl;

    std::cout << "=========== Zero Tensors ============" << std::endl;

    auto zero_tensor = torch::zeros({4, 4}, torch::TensorOptions(torch::kLong));
    std::cout << "4x4 Zero Tensor:\n" << zero_tensor << std::endl;

    std::cout << "=========== Randomly initialized Tensors ============" << std::endl;

    auto rand_tensor = torch::rand({4, 4});
    std::cout << "4x4 Random Tensor:\n" << rand_tensor << std::endl;

    std::cout << "=========== Manually initialized Tensors ============" << std::endl;

    auto manual_tensor = torch::tensor({2.2, 4.5, 5.6, 3.9});
    std::cout << "4x1 Manual Tensor:\n" << manual_tensor << std::endl;

    std::cout << "=========== Tensor Size checking ============" << std::endl;

    std::cout << "Manual Tensor Size: " << manual_tensor.element_size() << std::endl;
    std::cout << "Random Tensor Size: " << rand_tensor.element_size() << std::endl;

    std::cout << "=========== Tensors from previous Tensor ============" << std::endl;

    auto new_tensor = torch::zeros({4, 4}, torch::TensorOptions(torch::kDouble));
    std::cout << "New tensor:\n" << new_tensor << std::endl;

    auto new_tensor_from_old = torch::rand_like(new_tensor, torch::TensorOptions(torch::kFloat));
    std::cout << "New tensor from previous one:\n" << new_tensor_from_old << std::endl;

    std::cout << "=========== Tensor Addition ============" << std::endl;

    auto x = torch::rand({4, 4});
    auto y = torch::rand({4, 4});

    std::cout << "Method 1: " << x + y << std::endl;
    std::cout << "Method 2: " << torch::add(x, y) << std::endl;

    auto result = torch::empty({4, 4});
    torch::add_out(x, y, result);
    std::cout << "Method 3 [Provide OP tensor]: " << result << std::endl;

    std::cout << "Method 4 [In Place]" << y.add_(x) << std::endl;

    std::cout << "=========== Tensor indexing ============" << std::endl;

    auto idx_tensor = torch::rand({4, 4});
    std::cout << "idx_tensor[:, 1]:\n" << idx_tensor.index({at::indexing::Slice(), 1}) << std::endl;

    std::cout << "=========== Tensor Resizing ============" << std::endl;

    auto tensor1 = torch::randn({4, 4});

    auto tensor2 = tensor1.view(16);
    auto tensor3 = tensor1.view({-1, 8});

    std::cout << "Resized shapes:\n";
    std::cout << tensor1.element_size() << tensor2.element_size() << tensor3.element_size() << std::endl;

    std::cout << "=========== Grab the element from single Element Tensor ============" << std::endl;

    auto single_tensor = torch::randn(1);
    std::cout << "tensor:\n" << single_tensor << std::endl;
    std::cout << "tensor.item():\n" << single_tensor.item<float>() << std::endl;
}
