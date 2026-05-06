import matplotlib.pyplot as plt

loss_values = [1.1058, 1.0952, 1.0939, 1.0954, 1.0931, 1.0888, 1.0854, 1.0836]
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs, loss_values, linewidth=3)
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.savefig("training_loss_plot.png")
plt.close()


loss_vals_hybrid = [1.1388, 1.1107, 1.1020, 1.1038, 1.1064, 1.1058, 1.1018, 1.0964]
epochs = range(1, len(loss_vals_hybrid) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs, loss_vals_hybrid, linewidth=3)
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.savefig("training_loss_plot_hybrid.png")
plt.close()
