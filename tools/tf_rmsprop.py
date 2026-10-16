import torch

class RMSpropTF(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, rho=0.9, eps=1e-7):
        super().__init__(params, dict(lr=lr, rho=rho, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss=None

        if closure is not None:
            with torch.enable_grad():
                loss=closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("RMSpropTF does not support sparse gradients")

                grad=p.grad
                state=self.state[p]

                if "square_avg" not in state:
                    state["square_avg"]=torch.zeros_like(p)

                square_avg=state["square_avg"]
                square_avg.mul_(group["rho"]).addcmul_(grad, grad, value=1-group["rho"])
                denom=(square_avg+group["eps"]).sqrt()
                p.add_(-group["lr"]*grad/denom)

        return loss