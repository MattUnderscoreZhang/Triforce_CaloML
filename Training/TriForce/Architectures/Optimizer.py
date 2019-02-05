from torch.optim.optimizer import Optimizer, required
import copy


class LinearRampOptimizer(Optimizer):
    """ Uses the SGD with linearly ramped learning rate, as described in
        https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf?.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): max learning rate (required)
        ramp_factor (float): factor to ramp learning rate by (optional)
        ramp_batches (int): how many batches to ramp learning rate over (optional)
    """
    def __init__(self, params, lr=required, ramp_factor=10, ramp_batches=100):
        defaults = dict(lr=lr, ramp_factor=ramp_factor, ramp_batches=ramp_batches)
        super().__init__(params, defaults)
        self.target_lr = lr
        self.current_lr = lr/ramp_factor
        self.increase_lr = lr*(1-1/ramp_factor)/ramp_batches

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                p.data.add_(-self.current_lr, d_p)
        if self.current_lr < self.target_lr:
            self.current_lr += self.increase_lr
        return loss
