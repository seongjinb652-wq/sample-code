from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(w_array, b_array, loss_array)

ax.set_xlabel('w', size=16)
ax.set_ylabel('b', size=16)
ax.tick_params(labelsize=12)

plt.show()

loss_surface = []
w_surface = []
b_surface = []

for w_value in np.linspace(0, 20, 200):
    for b_value in np.linspace(-18, 22, 200):
        
        # Collect information about the loss function surface
        with torch.no_grad():
            model.weight[0] = w_value
            model.bias[0] = b_value

        outputs_predicted = model(inputs)
        loss = loss_fn(outputs_predicted, outputs)
        
        b_surface.append(b_value)
        w_surface.append(w_value)
        loss_surface.append(loss.item())


plt.close()

fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')

ax2.scatter(w_surface, b_surface, loss_surface, c = loss_surface, alpha = 0.02)
ax2.plot(w_array, b_array, loss_array, color='black')

ax2.set_xlabel('w')
ax2.set_ylabel('b')

plt.show()

# ---------------------------------------------------
model.reset_parameters()

max_number_of_epochs = 1000

loss_array = []
b_array = []
w_array = []

# Zero out the initial values
with torch.no_grad():
    model.weight[0] = 0
    model.bias[0] = 0
    
# 
outputs_predicted = model(inputs)
loss_value = loss_fn(outputs_predicted, outputs)
print("Before training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(model.weight.item(), model.bias.item(), loss_value))

print("")
print("Starting training")
print("")

# Start the training process
for i in range(max_number_of_epochs):
    
    # Update after every data point
    for (x_pt, y_pt) in zip(inputs, outputs):

        optimizer.zero_grad()

        y_pt_predicted = model(x_pt)

        loss = loss_fn(y_pt_predicted, y_pt)

        loss.backward()

        optimizer.step()
        
        w_array.append(model.weight.item())
        b_array.append(model.bias.item())
        loss_array.append(loss.item())

    # 
    if i > 0:
        avg_w = sum(w_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
        avg_b = sum(b_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
        avg_loss = sum(loss_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
        print("Epoch = {:2d}: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(i+1, avg_w, avg_b, avg_loss))

    if i > 1:
        average_loss_this_epoch = sum(loss_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
        average_loss_last_epoch = sum(loss_array[(i-2)*n_samples:(i-1)*n_samples]) / n_samples
        if abs(average_loss_this_epoch - average_loss_last_epoch) / average_loss_last_epoch < 0.001:
            break

print("")
print("Training finished after {} epochs".format(i+1))
print("")

plt.close()
plt.plot(loss_array)
plt.xlabel("Number of Updates", size=24)
plt.ylabel("Loss", size=24)
plt.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.show()

avg_w = sum(w_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
avg_b = sum(b_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples
avg_loss = sum(loss_array[(i-1)*n_samples:(i  )*n_samples]) / n_samples

print("After training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(avg_w, avg_b, avg_loss))

from mpl_toolkits.mplot3d import Axes3D
plt.close()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(w_array, b_array, loss_array)

ax.set_xlabel('w', size=16)
ax.set_ylabel('b', size=16)
ax.tick_params(labelsize=12)

plt.show()

plt.close()

fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')

ax2.scatter(w_surface, b_surface, loss_surface, c = loss_surface, alpha = 0.02)
ax2.plot(w_array, b_array, loss_array, color='black')

ax2.set_xlabel('w')
ax2.set_ylabel('b')

plt.show()
