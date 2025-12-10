#Intializing Tensors 
import torch
import numpy as np

#Tensors can initialized directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#or NumPy arrays 
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#or from another tensor 
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#you can create tensors of defined dimensions using the shape() command. Shape creates a tuple (a inmuttable data type) with a defined number of dimensions and then fill it with torch commands. 

shape = (2,3,) #creates a 2 x 3 tensor of values 
rand_tensor = torch.rand(shape) #creates a 2 x 3 tensor of random values 
ones_tensor = torch.ones(shape) #creates a 2 x 3 tensor of 1s 
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#you can inspect elements of tensors 
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#tensors are normalled stored on CPU, but you can run tensor functions on an accelerator (GPU) if available using the .to method 
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator()) #Not sure if this will work with my macbooks GPU or not 

#NumPy like slicing of a tensor. Uses Python-like indexing 
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

#torch.cat can be used to concatonate tensor together 
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#just playing around using indexing and altering entries in tensors 
#tensor[:,2] = 2
#print(tensor)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T) #matmul runs a matrix multiplication

y3 = torch.rand_like(y1) #creates a tensor of the same dimensions as the input tensor whith random numbers between 0 and 1 by default
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#if you have a tensor that is one value (for example you sum all values of a tensor together) you can change the data type to a python numerical using item()
agg = tensor.sum() #not quite sure how this is 12 instead of 9 since we change one of the entries to a 0???
agg_item = agg.item()
print(agg_item, type(agg_item))