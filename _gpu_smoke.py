import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("capability: sm_%d%d" % torch.cuda.get_device_capability(0))
x = torch.zeros(3).cuda()
y = (x + 1).sum().item()
print("KERNEL_OK", y)  # if this prints, real sm_120 kernels work
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import FAILED:", repr(e))
