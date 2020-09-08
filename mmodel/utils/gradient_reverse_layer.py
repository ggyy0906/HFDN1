from torch import nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, back_coeff):
        ctx.back_coeff = back_coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        back_coeff = ctx.back_coeff
        reverse_with_coeff = -grad_output*back_coeff
        return reverse_with_coeff, None

class GradReverseLayer(nn.Module):
    def __init__(self, coeff_fn):
        super().__init__()
        self.coeff_fn = coeff_fn
    
    def forward(self, x):
        x = GradReverse.apply(x, self.coeff_fn())
        return x

# if __name__ == "__main__":
#     import torch
#     a = GradReverseLayer()
#     x = torch.zeros([10], requires_grad=True)
#     y = torch.ones([10])
#     y_ = a(x)
#     loss = nn.BCELoss()(x,y)
#     loss.backward()


# import torch
# torch.nn.LeakyReLU

# class MyReLU(torch.autograd.Function):
#     """
#     We can implement our own custom autograd Functions by subclassing
#     torch.autograd.Function and implementing the forward and backward passes
#     which operate on Tensors.
#     """

#     @staticmethod
#     def forward(ctx, input):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         ctx.save_for_backward(input)
#         return input.clamp(min=0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         print("1111111111111111111")
#         return grad_input


# dtype = torch.float
# device = torch.device("cpu")
# # device = torch.device("cuda:0") # Uncomment this to run on GPU

# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Create random Tensors to hold input and outputs.
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)

# # Create random Tensors for weights.
# w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# learning_rate = 1e-6
# for t in range(500):
#     # To apply our Function, we use Function.apply method. We alias this as 'relu'.
#     relu = GradReverse.apply

#     grl = GradReverseLayer()

#     # Forward pass: compute predicted y using operations; we compute
#     # ReLU using our custom autograd operation.
#     y_pred = grl(x.mm(w1)).mm(w2)

#     # Compute and print loss
#     loss = (y_pred - y).pow(2).sum()
#     if t % 100 == 99:
#         print(t, loss.item())

#     # Use autograd to compute the backward pass.
#     loss.backward()

#     # Update weights using gradient descent
#     with torch.no_grad():
#         w1 -= learning_rate * w1.grad
#         w2 -= learning_rate * w2.grad

#         # Manually zero the gradients after updating weights
#         w1.grad.zero_()
#         w2.grad.zero_()