
from Model import model
# get data
x = sample_features.astype(np.float32)
# sample_features should contain flattened image features
x = x / 255.0
x = x.reshape(1, 784)
#  Define structure
nn = model(
    structure=[784, 128, 64, 26],
    base_address="./"
)

# 4. Run prediction
# This will output the prediction and confidence to prediction.txt
nn.run("./model.pkl", x)
