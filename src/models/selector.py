import os

from collections import OrderedDict

from collections import OrderedDict
import torch


def load_state_dict(net, orig_state_dict):
    # 如果传入的是模型对象（如 torch.nn.Sequential），提取其 state_dict
    if isinstance(orig_state_dict, torch.nn.Module):
        orig_state_dict = orig_state_dict.state_dict()

    # 处理多层嵌套的 state_dict（例如完整检查点文件）
    if isinstance(orig_state_dict, dict):
        # 合并 "state_dict" 和 'state_dict' 键的处理
        while 'state_dict' in orig_state_dict or "state_dict" in orig_state_dict:
            orig_state_dict = orig_state_dict.get('state_dict', orig_state_dict.get("state_dict"))

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        # 直接使用 in 进行键检查（无需 .keys()）
        if k in orig_state_dict:
            new_state_dict[k] = orig_state_dict[k]
        elif any(suffix in k for suffix in ['running_mean_noisy', 'running_var_noisy', 'num_batches_tracked_noisy']):
            # 处理带 _noisy 后缀的特殊参数
            base_key = k[:-6]
            new_state_dict[k] = orig_state_dict[base_key].clone().detach()
        else:
            # 保留网络原有参数（适用于新增层）
            new_state_dict[k] = v

    # 严格模式加载（验证参数完整性）
    net.load_state_dict(new_state_dict, strict=False)


if __name__ == '__main__':

    import torch
    from torchsummary import summary
    import random
    import time

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    support_x_task = torch.autograd.Variable(torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1))

    t0 = time.time()
    model = select_model('CIFAR10', model_name='WRN-16-2')
    output, act = model(support_x_task)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))