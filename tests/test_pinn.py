import torch
from physai.pinn import PINN
from physai.physics import dy_dx_equals_y

def test_loss_nonnegative():
    x = torch.tensor([[0.0], [1.0]], requires_grad=True)
    model = PINN([1,10,1])
    loss = model.physics_loss(x, lambda x,y: dy_dx_equals_y(x, model))
    assert loss.item() >= 0

if __name__ == '__main__':
    test_loss_nonnegative()
    print('Test passed!')
