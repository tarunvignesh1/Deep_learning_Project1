import torch as t
x = t.rand(5, 3)
print(x)

train_on_gpu = t.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')