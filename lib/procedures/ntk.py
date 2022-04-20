import numpy as np
import torch


def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network


def get_ntk_n(xloader, networks, recalbn=0, train_mode=False, num_batch=-1):
    device = torch.cuda.current_device()
    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    # 建立grads list，將裡面有數量為networks list長度的空list
    grads = [[] for _ in range(len(networks))]
    # xloader 內有 inputs和targets
    for i, (inputs, targets) in enumerate(xloader):
        # num_batch 預設為-1
        if num_batch > 0 and i >= num_batch: break
        # 將inputs放入gpu
        inputs = inputs.cuda(device=device, non_blocking=True)
        # 對networks list內的每個network
        for net_idx, network in enumerate(networks):
            # 將network的梯度歸零
            network.zero_grad()
            # inputs_會將梯度疊加給原始inputs
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            # logit 是inputs_作為輸入的netowrk輸出
            logit = network(inputs_)
            # 若logit 是tuple的話
            if isinstance(logit, tuple):
                # 則logit 將變成logit tuple第二個元素
                logit = logit[1]  # 201 networks: return features and logits
            # 將每個inputs_送進network中，並將梯度輸出放在logit list內
            for _idx in range(len(inputs_)):
                # 對於在inputs_中的每個input，傳入和輸出一樣shape且全為1的矩陣，可得到所有子結點的梯度
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                # 建立梯度list
                grad = []
                # 對所有netowrk的參數
                for name, W in network.named_parameters():
                    # 將權重梯度append進grad中
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                # 再將grad放進grads list中
                grads[net_idx].append(torch.cat(grad, -1))
                # 將network梯度歸零
                network.zero_grad()
                # 清空cache
                torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds
