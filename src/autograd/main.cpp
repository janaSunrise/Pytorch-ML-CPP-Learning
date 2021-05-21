#include <torch/torch.h>

#include <iostream>

using namespace torch::autograd;

void basic_autograd_operations() {
    std::cout << "====== Basic autograd operations ======" << std::endl;

    // Create a tensor and set ``torch::requires_grad()`` to track computation with it
    auto x = torch::ones({2, 2}, torch::requires_grad());
    std::cout << x << std::endl;

    // Do a tensor operation:
    auto y = x + 2;
    std::cout << y << std::endl;

    // ``y`` was created as a result of an operation, so it has a ``grad_fn``.
    std::cout << y.grad_fn()->name() << std::endl;

    // Do more operations on ``y``
    auto z = y * y * 3;
    auto out = z.mean();

    std::cout << z << std::endl;
    std::cout << z.grad_fn()->name() << std::endl;
    std::cout << out << std::endl;
    std::cout << out.grad_fn()->name() << std::endl;

    // ``.requires_grad_( ... )`` changes an existing tensor's ``requires_grad`` flag in-place.
    auto a = torch::randn({2, 2});
    a = ((a * 3) / (a - 1));
    std::cout << a.requires_grad() << std::endl;

    a.requires_grad_(true);
    std::cout << a.requires_grad() << std::endl;

    auto b = (a * a).sum();
    std::cout << b.grad_fn()->name() << std::endl;

    // Let's backprop now. Because ``out`` contains a single scalar, ``out.backward()``
    // is equivalent to ``out.backward(torch::tensor(1.))``.
    out.backward();

    // Print gradients d(out)/dx
    std::cout << x.grad() << std::endl;

    // Now let's take a look at an example of vector-Jacobian product:
    x = torch::randn(3, torch::requires_grad());

    y = x * 2;
    while (y.norm().item<double>() < 1000) {
        y = y * 2;
    }

    std::cout << y << std::endl;
    std::cout << y.grad_fn()->name() << std::endl;

    // If we want the vector-Jacobian product, pass the vector to ``backward`` as argument:
    auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
    y.backward(v);

    std::cout << x.grad() << std::endl;

    std::cout << x.requires_grad() << std::endl;
    std::cout << x.pow(2).requires_grad() << std::endl;

    {
        torch::NoGradGuard no_grad;
        std::cout << x.pow(2).requires_grad() << std::endl;
    }

    std::cout << x.requires_grad() << std::endl;
    y = x.detach();

    std::cout << y.requires_grad() << std::endl;
    std::cout << x.eq(y).all().item<bool>() << std::endl;
}

void compute_higher_order_gradients() {
    std::cout << "====== Computing higher-order gradients in C++ ======" << std::endl;

    auto model = torch::nn::Linear(4, 3);

    auto input = torch::randn({3, 4}).requires_grad_(true);
    auto output = model(input);

    // Calculate loss
    auto target = torch::randn({3, 3});
    auto loss = torch::nn::MSELoss()(output, target);

    // Use norm of gradients as penalty
    auto grad_output = torch::ones_like(output);
    auto gradient = torch::autograd::grad(
        {output}, {input}, /*grad_outputs=*/{grad_output}, /*create_graph=*/true
    )[0];
    auto gradient_penalty = torch::pow((gradient.norm(2, /*dim=*/1) - 1), 2).mean();

    // Add gradient penalty to loss
    auto combined_loss = loss + gradient_penalty;
    combined_loss.backward();

    std::cout << input.grad() << std::endl;
}

int main() {
    std::cout << std::boolalpha;

    // Run the basic autograd operations
    basic_autograd_operations();
    std::cout << std::endl;

    // Computation of the gradients
    compute_higher_order_gradients();
    std::cout << std::endl;
}
