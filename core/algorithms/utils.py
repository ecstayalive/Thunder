from torch.nn import Module


def soft_update(target_net: Module, net: Module, tau: float = 0.005) -> None:
    for target_net_param, net_param in zip(target_net.parameters(), net.parameters()):
        target_net_param.data.copy_(
            tau * net_param.data + (1 - tau) * target_net_param.data
        )


def hard_update(target_net: Module, net: Module) -> None:
    for target_net_param, net_param in zip(target_net.parameters(), net.parameters()):
        target_net_param.data.copy_(net_param.data)
