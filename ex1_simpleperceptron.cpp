#include <torch/torch.h>
#include <iostream>

struct Net : torch::nn::Module {
	Net(int64_t N, int64_t M)
			:linear(register_module("linear", torch::nn::Linear(N, M)))
	    {
			another_bias = register_parameter("b", torch::randn(M));
		}

	torch::Tensor forward(torch::Tensor input) 
	{
		return linear(input) + another_bias;
	}

		torch::nn::Linear linear;
		torch::Tensor another_bias;

};

int main()
{
	Net net(4, 5); //Define Network 

/*	for (const auto& p : net.parameters()) {

		std::cout << p << std::endl; //print all weights and bias

	} */


	for (const auto& pair : net.named_parameters()) {

		std::cout << pair.key() << ": " << pair.value() << std::endl;

	}

	torch::Tensor input_tensor = torch::ones({2,4}) ;

	std::cout << "input_tensor : " << input_tensor << std::endl; 

	std::cout << net.forward(input_tensor) << std::endl;
	
}