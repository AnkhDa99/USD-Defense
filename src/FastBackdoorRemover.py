# import autograd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.autograd as autograd


class FastBackdoorRemover(nn.Module):
    def __init__(self, args, device, model, criterion):
        super().__init__()
        self.model = model
        self.device = device
        self.criterion = criterion

        # 动态正则化参数
        self.alpha = args.alpha  # 梯度正则强度
        self.beta = args.beta  # 迹正则强度
        self.gamma = args.gamma  # EWC强度
        self.iter_gap = 5  # 迹计算间隔

        # 优化器配置
        self.optimizer = torch.optim.SGD(model.parameters(),
                                         lr=args.lr,
                                         momentum=0.95,
                                         weight_decay=1e-4)

        # 知识保留系统
        self.param_anchors = {}
        self.fisher_info = {}

        # 初始化锚点
        self._init_anchors()

    def _init_anchors(self):
        """初始化参数锚点"""
        for name, param in self.model.named_parameters():
            self.param_anchors[name] = param.data.clone()
            self.fisher_info[name] = torch.zeros_like(param.data)

    def update_anchors(self, clean_loader, num_batches=10):
        """使用干净样本更新锚点"""
        self.model.eval()
        fisher_counts = {name: torch.zeros_like(p) for name, p in self.fisher_info.items()}

        for i, (inputs, targets) in enumerate(clean_loader):
            if i >= num_batches: break

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 计算梯度平方(Fisher信息)
            grads = torch.autograd.grad(loss, self.model.parameters())
            for (name, _), grad in zip(self.model.named_parameters(), grads):
                fisher_counts[name] += grad.pow(2)

        # 更新Fisher信息
        for name in self.fisher_info:
            self.fisher_info[name] = fisher_counts[name] / num_batches
            self.param_anchors[name] = self.model.state_dict()[name].clone()

    def compute_regularization(self):
        """计算混合正则项"""
        reg_loss = 0.0

        # 1. 梯度范数正则 (持续作用)
        active_params = [p for p in self.model.parameters() if p.requires_grad]
        if active_params:
            grads = torch.autograd.grad(self.ce_loss, active_params, create_graph=True)
            grad_norm = torch.cat([g.flatten() for g in grads]).norm(2)
            reg_loss += self.alpha * grad_norm

        # 2. 迹估计正则 (周期性计算)
        if self.current_iter % self.iter_gap == 0:
            trace_loss = self.compute_trace()
            reg_loss += self.beta * trace_loss

        # 3. 锚点约束正则 (持续作用)
        anchor_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.param_anchors:
                anchor = self.param_anchors[name]
                fisher = self.fisher_info[name]
                anchor_loss += (fisher * (param - anchor).pow(2)).sum()
        reg_loss += self.gamma * anchor_loss

        return reg_loss

    def compute_trace(self, num_vectors=5):
        """高效迹估计"""
        # 使用随机投影法
        output = F.log_softmax(self.outputs, dim=1)
        log_likelihood = output[:, self.targets].mean()

        # 一阶梯度
        grads = autograd.grad(log_likelihood, self.model.parameters(), create_graph=True)

        # 随机向量法估计迹
        trace_est = 0.0
        for _ in range(num_vectors):
            v = [torch.randn_like(p) for p in self.model.parameters()]
            Hv = autograd.grad(grads, self.model.parameters(), grad_outputs=v, retain_graph=True)   ## 修改
            trace_est += sum(torch.sum(h * v_elem) for h, v_elem in zip(Hv, v))

        return trace_est / num_vectors

    def forward_backward_update(self, inputs, targets, iteration):
        self.current_iter = iteration
        self.targets = targets

        # 前向传播
        self.optimizer.zero_grad()
        self.outputs = self.model(inputs)
        self.ce_loss = self.criterion(self.outputs, targets)

        # 计算正则项
        reg_loss = self.compute_regularization()

        # 组合损失
        total_loss = self.ce_loss + reg_loss
        total_loss.backward()

        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # 参数更新
        self.optimizer.step()

        return total_loss, self.outputs