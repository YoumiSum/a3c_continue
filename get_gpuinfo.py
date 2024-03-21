import pycuda.driver as cuda

cuda.init()
num_gpus = cuda.Device.count()

for i in range(num_gpus):
    gpu = cuda.Device(i)
    print("GPU %d: %s" % (i, gpu.name()))
