import config 

x = np.float32(np.random.uniform(0, 10, [n_samples, 1]))
y = np.float32(np.array([w_gen * (x + np.random.normal(loc=mean_gen, scale=std_gen, size=None)) + b_gen for x in x]))

plt.close()
plt.plot(x, y, 'go')
plt.xlabel("x", size=24)
plt.ylabel("y", size=24)
plt.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.show()

x = torch.from_numpy(x) if not torch.is_tensor(x) else x
y = torch.from_numpy(y) if not torch.is_tensor(y) else y

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.nn.Linear(1, 1, bias = True, device = device)

# 
with torch.no_grad():   # 1
    model.weight[0] = 0
    model.bias[0] = 0

loss_fn = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

model.reset_parameters()

max_number_of_epochs = 1000

loss_array = []
b_array = []
w_array = []

inputs = x.to(device)
outputs = y.to(device)

outputs_predicted = model(inputs)
loss_value = loss_fn(outputs_predicted, outputs)
print("Before training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(model.weight.item(), model.bias.item(), loss_value))

print("")
print("Starting training")
print("")

for i in range(max_number_of_epochs):
    
    
    optimizer.zero_grad()    #1 

    outputs_predicted = model(inputs)
    loss = loss_fn(outputs_predicted, outputs)

    loss.backward()          #2 
    optimizer.step()         #3

    w_array.append(model.weight.item())
    b_array.append(model.bias.item())
    loss_array.append(loss.item())

    if (i + 1) % 5 == 0:
        print("Epoch = {:2d}: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(i+1, model.weight.item(), model.bias.item(), loss.item()))

    if loss.item() < 0.001:  # 오차가 0.001보다 작아지면
        print("충분히 학습되었습니다. 종료합니다.")
        break

print("")
print("Training finished after {} epochs".format(i+1))
print("")

print("After training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(model.weight.item(), model.bias.item(), loss.item()))

plt.close()
plt.plot(loss_array)
plt.xlabel("Epoch", size=24)
plt.ylabel("Loss", size=24)
plt.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.show()
