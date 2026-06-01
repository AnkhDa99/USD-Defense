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
    """FIP (EWC + Trace) with SI-style clean retention.

    - Keeps original interfaces and logging.

    - Adds SI buffers and penalty to preserve clean accuracy while removing backdoor.

    """
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

        # ---- SI (Synaptic Intelligence) for Clean Retention ----
        self.si_c = float(getattr(self.args, 'si_c', 0.1))        # strength of SI penalty
        self.si_epsilon = float(getattr(self.args, 'si_epsilon', 1e-3))
        self.si_update_gap = int(getattr(self.args, 'si_update_gap', 5))  # consolidate every k iters

        # Initialize SI state; final snapshot will be set at register_ewc_params()
        self.p_task_start = {n: p.data.clone() for n,p in self.model.named_parameters() if p.requires_grad}
        self.p_old = {n: p.data.clone() for n,p in self.model.named_parameters() if p.requires_grad}
        self.W = {n: p.data.clone().zero_() for n,p in self.model.named_parameters() if p.requires_grad}
        self.omega = {n: p.data.clone().zero_() for n,p in self.model.named_parameters() if p.requires_grad}

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
            # collect true-class log-prob per sample
            log_likelihoods.append(output[torch.arange(output.size(0)), target])

        log_likelihood = torch.cat(log_likelihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [name for name, param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            _buff_param_name = _buff_param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name + '_estimated_fisher', (param.data.clone() ** 2))

    ## Resgister using clean validation images
    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params_initial(dataset, batch_size, num_batches)
        self._update_mean_params()

        # ---- SI reset: take purification start as reference for clean retention ----
        self.p_task_start = {n: p.data.clone() for n,p in self.model.named_parameters() if p.requires_grad}
        self.p_old = {n: p.data.clone() for n,p in self.model.named_parameters() if p.requires_grad}
        for n in self.W: 
            self.W[n].zero_()
        for n in self.omega: 
            self.omega[n].zero_()

    ## EWC/FIP Regularization Loss
    def _compute_reg_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, f'{_buff_param_name}_estimated_mean')
                estimated_fisher = getattr(self.model, f'{_buff_param_name}_estimated_fisher')
                loss_consolidation = (estimated_fisher * (param - estimated_mean) ** 2).sum()
                losses.append(loss_consolidation)
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0

    # ---- SI helpers ----
    def _compute_si_reg_loss(self):
        if self.si_c <= 0:
            return 0.0
        reg = 0.0
        for n, p in self.model.named_parameters():
            if not p.requires_grad: 
                continue
            d_ref = (p - self.p_task_start[n])
            reg = reg + (self.omega[n] * d_ref.pow(2)).sum()
        return self.si_c * reg

    def _si_accumulate_W(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    # advance p_old even if no grad
                    self.p_old[n] = p.data.clone()
                    continue
                delta = (p.data - self.p_old[n])
                self.W[n].add_(-p.grad * delta)
                self.p_old[n] = p.data.clone()

    def _si_consolidate_online(self):
        if self.si_c <= 0:
            return
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                denom = (p.data - self.p_task_start[n]).pow(2) + self.si_epsilon
                self.omega[n].add_( self.W[n] / denom )
                self.W[n].zero_()

    def get_trace_loss(self, outputs, target, hi=20):
        output = F.log_softmax(outputs, dim=1)
        log_liklihoods = output[torch.arange(output.size(0)), target]
        log_likelihood = log_liklihoods.mean()
        Fv = AG.grad(log_likelihood, self.model.parameters(), create_graph=True)

        niters = hi
        V = []
        for _ in range(niters):
            V_i = [torch.randn_like(p, device=self.device) for p in self.model.parameters()]
            V.append(V_i)

        trace = []
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
        reg_loss = self._compute_reg_loss(self.weight)

        # Trace smoothing (FIP idea)
        if iteration % self.iter_gap == 0:
            trace_loss = self.get_trace_loss(outputs, target)
        else:
            trace_loss = 0.0

        # SI retention
        si_loss = self._compute_si_reg_loss()

        ## Total Loss
        loss = ce_loss + self.reg_F * trace_loss + reg_loss + si_loss
        loss.backward()
        self.optimizer.step()

        # ---- SI updates after stepping ----
        self._si_accumulate_W()
        if (self.si_update_gap > 0) and (iteration % self.si_update_gap == 0):
            self._si_consolidate_online()

        return loss, outputs

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
