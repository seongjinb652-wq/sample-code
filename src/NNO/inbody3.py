model.reset_parameters()
max_number_of_epochs = 1000

loss_array = []
b_array = []
w_array = []

with torch.no_grad():
    model.weight[0] = 0
    model.bias[0] = 0
    
outputs_predicted = model(inputs)
loss_value = loss_fn(outputs_predicted, outputs)
print("Before training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(model.weight.item(), model.bias.item(), loss_value))

print("")
print("Starting training")
print("")

batch_size = 32
# num_batches_in_epoch = FIXME
num_batches_in_epoch = len(inputs) // batch_size

# Start the training process
for i in range(max_number_of_epochs):

    for j in range(num_batches_in_epoch):
        # batch_start = FIXME
        batch_start = j * batch_size
        # batch_end = FIXME
        batch_end = (j + 1) * batch_size

        optimizer.zero_grad()

        outputs_predicted = model(inputs[batch_start:batch_end])

        loss = loss_fn(outputs_predicted, outputs[batch_start:batch_end])

        loss.backward()

        optimizer.step()

        w_array.append(model.weight.item())
        b_array.append(model.bias.item())
        loss_array.append(loss.item())

    if i > 0:
        avg_w = sum(w_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
        avg_b = sum(b_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
        avg_loss = sum(loss_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
        print("Epoch = {:2d}: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(i+1, avg_w, avg_b, avg_loss))

    if i > 1:
        average_loss_this_epoch = sum(loss_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
        average_loss_last_epoch = sum(loss_array[(i-2)*num_batches_in_epoch:(i-1)*num_batches_in_epoch]) / num_batches_in_epoch
        if abs(average_loss_this_epoch - average_loss_last_epoch) / average_loss_last_epoch < 0.001:
            break

print("")
print("Training finished after {} epochs".format(i+1))
print("")

avg_w = sum(w_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
avg_b = sum(b_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch
avg_loss = sum(loss_array[(i-1)*num_batches_in_epoch:(i  )*num_batches_in_epoch]) / num_batches_in_epoch

print("After training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(avg_w, avg_b, avg_loss))


plt.close()
plt.plot(loss_array)
plt.xlabel("Number of Updates", size=24)
plt.ylabel("Loss", size=24)
plt.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.show()

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

