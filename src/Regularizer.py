import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader
import torch.autograd as AG

# 在 Regularizer_ultra.py 中新增一个类

class SI_Regularizer:
    def __init__(self, args, device, model, crit, c=0.1, epsilon=0.001): # c是正则化强度
        self.model = model
        self.device = device
        self.crit = crit
        self.args = args
        self.c = c  # SI的正则化系数
        self.epsilon = epsilon # 防止除以0的小量

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.95)
        
        # SI需要存储的变量
        self.W = {name: p.data.clone().zero_() for name, p in self.model.named_parameters() if p.requires_grad}
        self.p_old = {name: p.data.clone() for name, p in self.model.named_parameters() if p.requires_grad}
        self.omega = {name: p.data.clone().zero_() for name, p in self.model.named_parameters() if p.requires_grad}

    # SI不需要像EWC那样的预计算步骤 (register_ewc_params可以移除)

    def _compute_reg_loss(self):
        """计算SI的正则化损失"""
        losses = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                _p_old = self.p_old[name]
                _omega = self.omega[name]
                # 核心公式: Ω * (θ - θ_old)^2
                losses.append((_omega * (param - _p_old) ** 2).sum())
        return self.c * sum(losses)

    def _update_omega(self):
        """在一次完整的训练迭代后更新重要性参数 omega"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad = param.grad.data
                p_new = param.data
                p_old = self.p_old[name]
                
                # 更新 W (梯度对参数变化的贡献)
                self.W[name].add_(-grad * (p_new - p_old))
                # 更新 p_old 为当前参数，为下次计算做准备
                self.p_old[name] = p_new.clone()

    def on_task_finish(self): # <--- 在每个净化阶段结束后调用
        """在一个任务完成后，将累积的W更新到最终的重要性omega中"""
        for name, _ in self.model.named_parameters():
             if name in self.omega:
                p_new = self.p_old[name]
                # 假设任务A的参数为p_start, 任务B结束后为p_end
                # delta_p = p_end - p_start
                # 这里简化处理，直接用W
                delta_p = p_new - getattr(self, f'{name}_task_start_param', p_new) # 需要保存任务开始时的参数
                self.omega[name].add_(self.W[name] / (delta_p ** 2 + self.epsilon))
                # 为新任务重置W
                self.W[name].zero_()
                # 保存新任务的初始参数
                setattr(self, f'{name}_task_start_param', p_new.clone())


    def forward_backward_update(self, input_s, target, iteration):
        self.optimizer.zero_grad()
        outputs = self.model(input_s)
        ce_loss = self.crit(outputs, target)
        reg_loss = self._compute_reg_loss() # 使用SI的正则化损失
        
        loss = ce_loss + reg_loss
        loss.backward()
        self.optimizer.step()
        
        # SI需要在梯度更新后，再更新omega
        self._update_omega() 

        return loss, outputs

class CDA_Regularizer:

    def __init__(self, args, device, model, crit, lr=0.002, reg_F=0.01, weight=5):
        self.model = model
        self.device = device
        self.weight = weight
        self.reg_F = args.reg_F
        self.crit = crit
        self.args = args
        self.iter_gap = 5
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.95)
        print("Regularizer Coffieicients:", self.iter_gap, self.reg_F, self.weight)

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params_initial(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_likelihoods = []
        for i, (inputs, target) in enumerate(dl):
            if i > num_batch:
                break
            inputs, target = inputs.cuda(), target.cuda()
            output = F.log_softmax(self.model(inputs), dim=1)
            log_likelihoods.append(output[:, target])

        log_likelihood = torch.cat(log_likelihoods).mean()
        # print("log likelihood:", log_likelihood)

        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [name for name, param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            _buff_param_name = _buff_param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)

    ## Resgister using clean validation images
    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params_initial(dataset, batch_size, num_batches)
        self._update_mean_params()

    ## Regularization Loss
    def _compute_reg_loss(self, weight, exemption_mask=None):
        try:
            losses = []
            # 如果没有提供豁免面具，则创建一个全为1的默认面具（即标准FIP）
            if exemption_mask is None:
                exemption_mask = {name: 1.0 for name, _ in self.model.named_parameters()}

            for param_name, param in self.model.named_parameters():
                if param.requires_grad:
                    _buff_param_name = param_name.replace('.', '__')
                    
                    # 检查buffer是否存在，避免报错
                    mean_buffer_name = '{}_estimated_mean'.format(_buff_param_name)
                    fisher_buffer_name = '{}_estimated_fisher'.format(_buff_param_name)
                    if hasattr(self.model, mean_buffer_name) and hasattr(self.model, fisher_buffer_name):
                        estimated_mean = getattr(self.model, mean_buffer_name)
                        estimated_fisher = getattr(self.model, fisher_buffer_name)
                        
                        # 获取该参数的豁免系数（0或1）
                        mask_value = exemption_mask.get(param_name, 1.0)
                        
                        loss_consolidation = (estimated_fisher * (param - estimated_mean) ** 2).sum()
                        
                        # 将豁免系数应用到损失上
                        losses.append(mask_value * loss_consolidation)

            if not losses:
                return torch.tensor(0.0).to(self.device)

            return (weight / 2) * sum(losses)

        except AttributeError:
            return torch.tensor(0.0).to(self.device)

    def get_trace_loss(self, outputs, target, hi=20):

        output = F.log_softmax(outputs, dim=1)
        log_liklihoods = output[:, target]
        log_likelihood = log_liklihoods.mean()
        Fv = AG.grad(log_likelihood, self.model.parameters(), create_graph=True)

        # for V_i in V:
        #     # Hv = AG.grad(Fv, params, V_i, create_graph=True)

        niters = hi
        V = list()
        for _ in range(niters):
            # V_i = [torch.randint_like(p, high=2, device=device) for p in model.parameters()]
            # for V_ij in V_i:
            #     V_ij[V_ij == 0] = -1
            V_i = [torch.randn_like(p, device=self.device) for p in self.model.parameters()]
            V.append(V_i)

        trace = list()
        for V_i in V:
            this_trace = 0.0
            for Hv_, V_i_ in zip(Fv, V_i):
                this_trace = this_trace + torch.sum(Hv_ * V_i_)
                trace.append(this_trace)

        return sum(trace) / niters

    def forward_backward_update(self, input_s, target, iteration):
        self.optimizer.zero_grad()

        outputs = self.model(input_s)
        ce_loss = self.crit(outputs, target)
        reg_loss    = self._compute_reg_loss(self.weight)

        ### Backpropagate the loss
        if iteration % self.iter_gap == 0:
            trace_loss = self.get_trace_loss(outputs, target)
        else:
            trace_loss = 0

        ## Total Loss
        loss = ce_loss + self.reg_F * trace_loss   + reg_loss
        loss.backward()
        self.optimizer.step()

        return loss, outputs

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)