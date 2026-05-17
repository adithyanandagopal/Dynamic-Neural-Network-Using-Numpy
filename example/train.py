from Model import model

# 1. Define Structure
# Using MNIST data: 784 input pixels (28x28), 128 and 64 hidden nodes, 26 output classes
nn = model(
    structure=[784, 128, 64, 26],
    base_address="./"
)

# 2. Train model with specified parameters
nn.fit(
    train_data=r"C:\adi\projects\hand gesture\hand gesture\dataset\sign_mnist_train.csv",
    test_data=r"C:\adi\projects\hand gesture\hand gesture\dataset\sign_mnist_test.csv",
    epoch=15,
    alpha=0.01,
    batch_size=50
)
