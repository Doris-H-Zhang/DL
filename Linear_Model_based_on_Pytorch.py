import torch
x_data= torch.Tensor([[1.0], [2.0], [3.0]])
y_data= torch.Tensor([[2.0], [4.0], [6.0]])

#Design model
class LinearModel(torch.nn.Module):#model 由pytorch现成的计算块组成
    def _init_(self):
        super(LinearModel, self)._init_()
        self.linear=torch.nn.Linear(1,1)#构造对象 w和b：weight和bias
    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred
    
model=LinearModel

#Construct loss and optimizer
criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameter(), lr=0.01)

#Training cycle
for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss)

    optimizer.zero_grad()#梯度清零
    loss.backward()
    optimizer.step()#更新

#Output
print('w=',model.linear.weight.item())#item会把矩阵改成数值
print('b=',model.linear.bias.item())

#Test
x_test=torch.Tensor([[4.0]])
y_test=model(x_test)
print('y_pred=',y_test.data)