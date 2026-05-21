import torch.optim as optim

def get_optimizer(
        optimizer_name, 
        model_parametes,
        learining_rate
    ):

    if optimizer_name == "sgd":
        return optim.SGD(
            model_parametes,
            lr = learining_rate
        )
    
    elif optimizer_name == "momentum":
        return optim.SGD(
            model_parametes,
            lr = learining_rate,
            momentum = 0.9
        )
    
    elif optimizer_name == "adam":
        return optim.Adam(
            model_parametes,
            lr = learining_rate
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")