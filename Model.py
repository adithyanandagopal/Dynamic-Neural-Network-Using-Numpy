"""
================================================================================
NeuralNet from Scratch : Dynamic Neural Network using NumPy
================================================================================

DEPENDENCIES:
    pip install numpy pandas matplotlib

USAGE:
    1. Initialize model with your layer structure:
          nn = model(
                structure   = [input_size, hidden1, hidden2,hidden n, output_size],
                base_address = "path/to/save/folder/"   # include trailing slash
          )
          example: model(structure=[784, 128, 64, 24], base_address="C:/models/")
        choose number of hidden layers and neurons as per need.

        ReLu activation is inside hidden layers
        Sigmoid activation for output layer for 1 output neuron 
        softmax activation for outputlayer with >1 neurons

    2. Train:
          nn.fit(
                train_data = "path/to/train.csv",
                test_data  = "path/to/test.csv",
                epoch      = 10,
                alpha      = 0.01,       # learning rate
                batch_size = 64
          )

    3. Predict:
          nn.run("path/to/model.pkl", x)

CSV FORMAT:
    First column must be named 'label'
    Remaining columns = features
    example: label, pixel1, pixel2, ... pixelN

PREPROCESSING (user responsibility before passing data):
    x = x / 255.0              # normalize pixel values to 0.0 - 1.0
    x = x.reshape(1, n)        # flatten to (1, num_features)
    labels must be 0-indexed   # 0 to num_classes-1

WEIGHTS:
    Saved automatically after training as 'model.pkl'
    Location: base_address/model.pkl
================================================================================
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class model:
    def __init__(self,structure:list[int],base_address:str):
        self.structure = structure
        self.weights =[]
        self.bias = []
        self.lyrs = len(structure)
        self.ip = structure[0]
        self.op = structure[-1]
        
        self.base_address = base_address

        if self.op == 1:
            self.op_activation = self.sigmoid
        else :
            self.op_activation = self.softmax

            

    # required functions
    def relu(self,x):
        return np.maximum(0,x)
    
    def relu_der(self,x):
        return (x>0).astype(int)
    
    def softmax(self,a):
        alpha = np.exp(a - np.max(a, axis=-1, keepdims=True))
        return alpha / np.sum(alpha, axis=-1, keepdims=True)
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def one_hot_enc(self,label,j):
        y=np.zeros(self.op)
        y[label[j]]=1
        return y.reshape(1,self.op)

    def cross_entropy(self,z3,y):
        if self.op == 1:
            #binary cros entropy loss fn
            error = -(y * np.log(z3 + 1e-9) + (1-y) * np.log(1 - z3 + 1e-9))  # 1e-9 is added for numerical stabilty

        else:
            #multiclass
            error = -(np.sum(y*np.log(z3 + 1e-9)))
        return error
    

    
    def forward_pass(self,w,b,x):
        self.a = [] # raw outputs of each layer
        self.z = [] # after activation
        nw = len(w)
        for i in range(nw):
                raw = x @ w[i] + b[i]
                actv = self.relu(raw) if i!= nw-1  else self.op_activation(raw) #activated output
                self.a.append(raw)
                self.z.append(actv)
                x = actv

        return self.z[-1]
    
    def delta(self,y):
        Delta = []
        n = len(self.weights)
        dt = self.z[-1] - y
        Delta.append(dt)
        for i in range(1,n):
            dt = (dt @ self.weights[n-i].T)*self.relu_der(self.a[n-1-i])
            Delta.append(dt)
        return Delta[::-1]
        

    def fit(self,train_data:str,test_data:str,epoch:int,alpha:float,batch_size:int):
        assert epoch>0, "epoch must be greater than 0"
        assert 0<alpha<1, "alpha range -> (0,1)"
        assert batch_size>0," batch size cant be less than 1"
        #loading data
        try:
            df_train = pd.read_csv(train_data)
            df_test = pd.read_csv(test_data)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e}")

        train_y = df_train["label"].values
        train_x = df_train.drop('label',axis=1).values
        train_x = train_x.astype(np.float32) # DL standard float 32

        test_y = df_test['label'].values
        test_x = df_test.drop('label',axis = 1).values
        test_x = test_x.astype(np.float32)
        print("data loaded!.....")

        #initialisng initial weights and bias 

        for i in range(self.lyrs -1):
            ip_n = self.structure[i] #input neuron
            op_n = self.structure[i+1] #output neuron
            w = np.random.randn(ip_n,op_n)*np.sqrt(2/ip_n)
            b = np.zeros((1,op_n))
            
            self.weights.append(w)
            self.bias.append(b)
        n =len(self.weights)
        

        print("weights initialised!...")

        print("training.....")

        loss_history = []#
        acc_history = []#


        for i in range(epoch):
            total_loss = 0
            acc = 0
            for j in range(0,len(train_y),batch_size):
                loss = 0 #setting loss back to zero
                #  dw = np.zeros(len(self.weights))
                # error db = np.zeros(len(self.bias))
                dw = []
                db = []
                for k in range(n):
                    db.append(0)
                    dw.append(0)
                strt = j
                end = strt + batch_size

                x_batch = train_x[strt:end]
                y_batch = train_y[strt:end]


                size = len(y_batch)
                for k in range(len(y_batch)):

                    x = x_batch[k]
                    x = x.reshape(1,self.ip)

                    y = self.one_hot_enc(y_batch,k)

                    z3 = self.forward_pass(self.weights,self.bias,x)
                    
                    
                    loss +=self.cross_entropy(z3,y)  # later divide by batch size
                    #accuracy
                    if np.argmax(z3) == y_batch[k]:
                        acc +=1

                    #delta
                    delta = self.delta(y)
                    cur = x
                    for d in range(len(delta)):

                        dw[d] += cur.T @ delta[d]  # (784,1) @ (1,128) = (784,128)
                        cur =self.z[d]

                        db[d] += np.sum(delta[d],axis = 0,keepdims=True)

                    #calculaton part done 
                #updation
                for d in range(len(delta)):
                    self.weights[d] -= alpha * (dw[d]/size)
                    self.bias[d] -= alpha * (db[d]/size)

                
                total_loss += loss/size
            accuracy = acc*100/len(train_x)
            
            loss_history.append(total_loss)
            acc_history.append(accuracy)

            print(f"\n loss = {total_loss}")
            print(f"\n accuracy = {accuracy}%")
            print(f"\n-------{i+1} epoch done------")
        print("-------Training-------") 

        #claude
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)          # 1 row, 2 plots, first one
        plt.plot(loss_history, color='red')
        plt.title('Loss')
        plt.xlabel('Epoch')

        plt.subplot(1, 2, 2)          # second plot
        plt.plot(acc_history, color='green')
        plt.title('Accuracy %')
        plt.xlabel('Epoch')

        plt.tight_layout()
        plt.show()
        
        address = self.base_address + "model.pkl"
        with open(address, 'wb') as f:
            pickle.dump({'weights': self.weights, 'bias': self.bias}, f)
        print("saved!")

        acc = 0
        test_size = len(test_y)
        for i in range(test_size):
            x = test_x[i]
            x = x.reshape(1,self.ip).astype(np.float32)
            y = test_y[i]
            z3 = self.forward_pass(self.weights,self.bias,x)
            if np.argmax(z3) == y:
                acc +=1
        accuracy = (acc*100)/test_size
        print(f" test accuracy { accuracy}%")
        
    def run(self, model_weights:str, x,):
        with open(model_weights, 'rb') as f:
            params = pickle.load(f)
        weights = params['weights']
        bias    = params['bias']

        z3 = self.forward_pass(weights,bias,x)
        prediction = z3.argmax()
        confidence  = z3.max()*100
        print(f" prediction = {prediction} with {confidence} % confidence")
        

