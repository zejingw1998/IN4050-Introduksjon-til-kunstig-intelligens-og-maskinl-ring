import numpy as np
import matplotlib.pyplot as plt
import sklearn # This is only to generate a dataset


# Generating the dataset
from sklearn.datasets import make_blobs
X, t_multi = make_blobs(n_samples=[400, 400, 400, 400, 400], centers=[[0,1],[4,2],[8,1],[2,0],[6,0]], 
                  n_features=2, random_state=2024, cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5])


# Shuffling the dataset
indices = np.arange(X.shape[0])
rng = np.random.RandomState(2024)
rng.shuffle(indices)
indices[:10]



# Splitting into train, dev and test
X_train = X[indices[:1000],:]
X_val = X[indices[1000:1500],:]
X_test = X[indices[1500:],:]
t_multi_train = t_multi[indices[:1000]]
t_multi_val = t_multi[indices[1000:1500]]
t_multi_test = t_multi[indices[1500:]]



t2_train = t_multi_train >= 3
t2_train = t2_train.astype('int')
t2_val = (t_multi_val >= 3).astype('int')
t2_test = (t_multi_test >= 3).astype('int')



plt.figure(figsize=(8,6)) # You may adjust the size
plt.scatter(X_train[:, 0], X_train[:, 1], c=t_multi_train, s=10.0)
plt.title("Multi-class set")




plt.figure(figsize=(8,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=t2_train, s=10.0)
plt.title("Binary set")



def add_bias(X, bias):
    """X is a NxM matrix: N datapoints, M features
    bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
    Return a Nx(M+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias # Make a N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 



class NumpyClassifier():
    """Common methods to all Numpy classifiers --- if any"""



class NumpyLinRegClass(NumpyClassifier):

    def __init__(self, bias=-1):
        self.bias=bias

    def fit(self, X_train, t_train, lr = 0.1, epochs=10):
        """X_train is a NxM matrix, N data points, M features
        t_train is avector of length N,
        the target class values for the training data
        lr is our learning rate
        """

        if self.bias:
            X_train = add_bias(X_train, self.bias)

        (N, M) = X_train.shape

        self.w = np.zeros(M)

        for epoch in range(epochs):
            # print("Epoch", epoch)
            prediction = X_train @ self.w
            error = prediction - t_train
            self.w -= lr / N * X_train.T @ error

    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = add_bias(X, self.bias)
        ys = X @ self.w
        return ys > threshold



def accuracy(predicted, gold):
    return np.mean(predicted == gold)



cl = NumpyLinRegClass()
cl.fit(X_train, t2_train, epochs=3)
print("Accuracy on the validation set:", accuracy(cl.predict(X_val), t2_val))



def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a prediction of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Classify each meshpoint.
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=size) # You may adjust this

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='Paired')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")



plot_decision_regions(X_train, t2_train, cl)


learning_rates = [0.001, 0.01, 0.05]
epochs_list = [10, 100, 500]

best_accuracy = 0
best_LR = None
best_EPOCHS = None
best_CL = None

for lr in learning_rates:
    for epochs in epochs_list:
        CL = NumpyLinRegClass()
        CL.fit(X_train, t2_train, lr=lr, epochs=epochs)

        val_accuracy = accuracy(CL.predict(X_val), t2_val)
        print(f"Learning rate: {lr}, Epochs: {epochs}, Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_LR = lr
            best_EPOCHS = epochs
            best_CL = CL

print("Best learning rate:", best_LR)
print("Best epochs:", best_EPOCHS)
print("Best accuracy:", best_accuracy)

#Conclusion




class StandardScaler:
    """Scales each feature to mean 0 and standard deviation 1"""

    def fit(self, X):
        """Calculate the mean and standard deviation of each feature in X."""
        
        self.mean = np.mean(X,axis = 0)
        self.stddev = np.std(X,axis=0)
        self.stddev = np.where(self.stddev == 0, 1, self.stddev)

        return self

    def transform(self, X):
        """Scale X with the mean and standard deviation learned in fit()."""
        return (X - self.mean) / self.stddev
        

    def normalize(self, X):
        return self.fit(X).transform(X)



scaler = StandardScaler()

X_train_scaled = scaler.normalize(X_train)
X_val_scaled = scaler.transform(X_val)


print("Mean:", np.mean(X_train_scaled, axis=0))
print("Std:", np.std(X_train_scaled, axis=0))
print("Shape:", X_train_scaled.shape)


# Tuning with scaled data
learning_rates = [0.001, 0.01, 0.05]
epochs_list = [10, 100, 500]

best_accuracy_scaled = 0
best_LR_scaled = None
best_EPOCHS_scaled = None
best_CL_scaled = None

for lr in learning_rates:
    for epochs in epochs_list:
        CL = NumpyLinRegClass()
        CL.fit(X_train_scaled, t2_train, lr=lr, epochs=epochs)

        val_accuracy = accuracy(CL.predict(X_val_scaled), t2_val)
        print(f"Learning rate: {lr}, Epochs: {epochs}, Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy_scaled:
            best_accuracy_scaled = val_accuracy
            best_LR_scaled = lr
            best_EPOCHS_scaled = epochs
            best_CL_scaled = CL

print("Best learning rate:", best_LR_scaled)
print("Best epochs:", best_EPOCHS_scaled)
print("Best accuracy:", best_accuracy_scaled)

#Conclusion
#The mean is approximately 0 and the standard deviation is 1.
#This shows that the scaling worked correctly.


# Your code here (see Lecture 07 notebooks for help)

#My own code 

#Define the sigmoid and binary cross entropy

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


#Part a

def train_logistic_regression(X, y, learning_rate=0.1, epochs=500, lambda_=0.0):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    history = []
    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        data_loss = binary_cross_entropy(y, p)
        history.append(data_loss + lambda_ * np.sum(w ** 2))

        error = p - y
        dw = X.T @ error / n_samples
        dw += 2 * lambda_ * w
        db = np.mean(error)


        #Gradient descent
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b, np.array(history)

class NumpyLogRegClass:
    def __init__(self):
        self.w = None
        self.b = None 
        self.history = None 

    def fit(self, X_train, t_train, learning_rate=0.1, epochs=500, lambda_=0.0):
        self.w, self.b, self.history = train_logistic_regression(X_train, t_train, learning_rate=learning_rate, epochs=epochs, lambda_=lambda_)
        return self 

    def predict(self,X):
        o =  (sigmoid(X @ self.w + self.b) >= 0.5).astype(int)
        return o
#Part b
    def predict_probability(self,X):
        o = sigmoid(X @ self.w + self.b)
        return o

#Test 
CL = NumpyLogRegClass()
CL.fit(X_train_scaled, t2_train, learning_rate=0.1, epochs=500)

print("Validation Accuracy:", accuracy(CL.predict(X_val_scaled), t2_val))
print("Probabilities:", CL.predict_probability(X_val_scaled[:5]))

#Part c
train_loss_history = CL.history
train_accuracy_history = []

w_c = np.zeros(X_train_scaled.shape[1])

b_c = 0.0

for epoch in range(500):
    p_c = sigmoid(X_train_scaled @ w_c + b_c)
    predictions = (p_c >= 0.5).astype(int)
    train_accuracy = np.mean(predictions == t2_train)
    train_accuracy_history.append(train_accuracy)

    error = p_c - t2_train
    dw = X_train_scaled.T @ error / len(t2_train)
    db = np.mean(error)

    #Gradient descent
    w_c -= 0.1 * dw
    b_c -= 0.1 * db

print("First Training Loss:", train_loss_history[0])
print("Final Training Loss:", train_loss_history[-1])
print("First Training Accuracy:", train_accuracy_history[0])
print("Final Training Accuracy:", train_accuracy_history[-1])
#Part d

class NumpyLogRegClassExtended(NumpyLogRegClass):
    def __init__(self):
        super().__init__()
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.n_epochs_trained = 0

    def fit(self, X_train, t_train, learning_rate=0.1, epochs=500, lambda_=0.0, X_val=None, t_val=None, tol=None, n_epochs_no_update=5):
        if (X_val is None) != (t_val is None):
            raise ValueError("X_val and t_val must both be provided")

        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.n_epochs_trained = 0
        best_loss = float("inf")
        no_update = 0

        for epoch in range(epochs):
            p = sigmoid(X_train @ self.w + self.b)
            error = p - t_train
            dw = X_train.T @ error / n_samples + 2 * lambda_ * self.w
            db = np.mean(error)

            #Gradient descent
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            #Training loss and accuracy
            p_train = sigmoid(X_train @ self.w + self.b)
            train_loss = binary_cross_entropy(t_train, p_train) + lambda_ * np.sum(self.w ** 2)
            self.history.append(train_loss)
            self.accuracy_history.append(np.mean((p_train >= 0.5) == t_train))

            #Validation loss and accuracy
            if X_val is not None:
                p_val = sigmoid(X_val @ self.w + self.b)
                val_loss = binary_cross_entropy(t_val, p_val)
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(np.mean((p_val >= 0.5) == t_val))

            #Part e - Early stopping
            self.n_epochs_trained = epoch + 1
            monitored_loss = val_loss if X_val is not None else train_loss

            if tol is not None:
                if best_loss - monitored_loss > tol:
                    best_loss = monitored_loss
                    no_update = 0
                else:
                    no_update += 1

                if no_update >= n_epochs_no_update:
                    break

        self.history = np.array(self.history)
        self.accuracy_history = np.array(self.accuracy_history)
        self.val_loss_history = np.array(self.val_loss_history)
        self.val_accuracy_history = np.array(self.val_accuracy_history)

        return self


#Test Part d
CL_d = NumpyLogRegClassExtended()
CL_d.fit(X_train_scaled, t2_train, X_val=X_val_scaled, t_val=t2_val, learning_rate=0.1, epochs=500)

print("Training Loss:", CL_d.history[-1])
print("Training Accuracy:", CL_d.accuracy_history[-1])
print("Validation Loss:", CL_d.val_loss_history[-1])
print("Validation Accuracy:", CL_d.val_accuracy_history[-1])
#Part e

CL_e = NumpyLogRegClassExtended()
CL_e.fit(X_train_scaled, t2_train, X_val=X_val_scaled, t_val=t2_val, learning_rate=0.1, epochs=1000, tol=0.0001, n_epochs_no_update=5)

print("Epochs trained:", CL_e.n_epochs_trained)
print("Validation Accuracy:", CL_e.val_accuracy_history[-1])
print("Validation Loss:", CL_e.val_loss_history[-1])

#Part f

learning_rates = [0.01, 0.05, 0.1]
tolerances = [0.001, 0.0001, 0.00001]

best_log_accuracy = -1
best_log_model = None
best_log_config = None

datasets = [("Original", X_train, X_val), ("Scaled", X_train_scaled, X_val_scaled)]

for label, X_tr, X_v in datasets:
    for lr in learning_rates:
        for tol in tolerances:
            CL = NumpyLogRegClassExtended()
            CL.fit(X_tr, t2_train, X_val=X_v, t_val=t2_val, learning_rate=lr, epochs=1000, tol=tol, n_epochs_no_update=5)

            val_accuracy = CL.val_accuracy_history[-1]
            print(f"{label}, LR: {lr}, Tolerance: {tol}, Epochs: {CL.n_epochs_trained}, Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_log_accuracy:
                best_log_accuracy = val_accuracy
                best_log_model = CL
                best_log_config = (label, lr, tol)

print("Best configuration:", best_log_config)
print("Best accuracy:", best_log_accuracy)
print("Best epochs:", best_log_model.n_epochs_trained)

#Part g

#Training and Validation Loss
plt.figure(figsize=(8,5))
plt.plot(best_log_model.history, label="Training Loss")
plt.plot(best_log_model.val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


#Training and Validation Accuracy
plt.figure(figsize=(8,5))
plt.plot(best_log_model.accuracy_history, label="Training Accuracy")
plt.plot(best_log_model.val_accuracy_history, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

#Conclusion
#The best validation accuracy is 76.2% using scaled data.
#The training and validation losses decrease smoothly.
#The accuracy remains almost constant despite decreasing loss.
#No clear overfitting is observed.
#Early stopping reduces unnecessary training epochs.
#LR=0.1 and tol=0.001 achieve 76.2% accuracy in only 148 epochs.




# Your code here (see Lecture 07 theory notebook, section 10, and practice Exercises 1-6 for the binary classifiers)

#Multi-class classifiers

#One-vs-rest with Logistic Regression

class OneVsRest:
    def __init__(self):
        self.models = []
        self.classes = None

    def fit(self, X_train, t_train, learning_rate=0.1, epochs=500):
        self.classes = np.unique(t_train)
        self.models = []

        for class_label in self.classes:
            t_binary = (t_train == class_label).astype(int)

            model = NumpyLogRegClass()
            model.fit(X_train, t_binary, learning_rate=learning_rate, epochs=epochs)
            self.models.append(model)

        return self

    def predict_probability(self, X):
        probabilities = np.column_stack([model.predict_probability(X) for model in self.models])
        return probabilities

    def predict(self, X):
        probabilities = self.predict_probability(X)
        predictions = np.argmax(probabilities, axis=1)
        return self.classes[predictions]

#Training and Validation

CL_multi = OneVsRest()
CL_multi.fit(X_train_scaled, t_multi_train, learning_rate=0.1, epochs=500)

predictions = CL_multi.predict(X_val_scaled)
val_accuracy = accuracy(predictions, t_multi_val)

print("Validation Accuracy:", val_accuracy)
print("First 5 predictions:", predictions[:5])
print("First 5 true labels:", t_multi_val[:5])


#Hyperparameter Tuning

learning_rates = [0.01, 0.05, 0.1]
epochs_list = [100, 300, 500]

best_accuracy = 0
best_LR = None
best_EPOCHS = None
best_CL = None

for lr in learning_rates:
    for epochs in epochs_list:
        CL = OneVsRest()
        CL.fit(X_train_scaled, t_multi_train, learning_rate=lr, epochs=epochs)

        val_accuracy = accuracy(CL.predict(X_val_scaled), t_multi_val)
        print(f"Learning rate: {lr}, Epochs: {epochs}, Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_LR = lr
            best_EPOCHS = epochs
            best_CL = CL

print("Best Learning Rate:", best_LR)
print("Best Epochs:", best_EPOCHS)
print("Best Accuracy:", best_accuracy)


#Decision Boundaries

plot_decision_regions(X_train_scaled, t_multi_train, best_CL)
plt.show()

#Conclusion

#The Validation accuracy is 0.83
#The best learning rate is 0.1
#The best number of epochs is 500
#The best accuracy is 83%
#Higher learning rates and more epochs generally improve the accuracy
#The decision boundaries separate the five classes well, but some points are misclassified 




def softmax(z, axis=-1):
    shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=axis, keepdims=True)


def multiclass_cross_entropy(target, probabilities):
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return -np.sum(target * np.log(probabilities))

#Multinomial Logistic Regression

#My own code

class NumpySoftmaxClass:
    def __init__(self):
        self.W = None
        self.b = None
        self.history = []
        self.classes = None

    def fit(self, X_train, t_train, learning_rate=0.1, epochs=500):
        n_samples, n_features = X_train.shape
        self.classes = np.unique(t_train)
        n_classes = len(self.classes)

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.history = []

        Y = (t_train[:, None] == self.classes).astype(int)

        for epoch in range(epochs):
            z = X_train @ self.W + self.b
            p = softmax(z)

            error = p - Y
            dw = X_train.T @ error / n_samples
            db = np.mean(error, axis=0)

            #Gradient descent
            self.W -= learning_rate * dw
            self.b -= learning_rate * db

            #Training Loss
            p_new = softmax(X_train @ self.W + self.b)
            loss = multiclass_cross_entropy(Y, p_new) / n_samples
            self.history.append(loss)

        return self

    def predict_probability(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        probabilities = self.predict_probability(X)
        predictions = np.argmax(probabilities, axis=1)
        return self.classes[predictions]

#Test

CL_softmax = NumpySoftmaxClass()
CL_softmax.fit(X_train_scaled, t_multi_train, learning_rate=0.1, epochs=500)

print("Validation Accuracy:", accuracy(CL_softmax.predict(X_val_scaled), t_multi_val))
print("First 5 predictions:", CL_softmax.predict(X_val_scaled[:5]))
print("First 5 probabilities:", CL_softmax.predict_probability(X_val_scaled[:5]))

#Hyperparameter Tuning

learning_rates = [0.01, 0.05, 0.1]
epochs_list = [100, 300, 500]

best_accuracy_softmax = 0
best_LR_softmax = None
best_EPOCHS_softmax = None
best_CL_softmax = None

for lr in learning_rates:
    for epochs in epochs_list:
        CL = NumpySoftmaxClass()
        CL.fit(X_train_scaled, t_multi_train, learning_rate=lr, epochs=epochs)

        val_accuracy = accuracy(CL.predict(X_val_scaled), t_multi_val)
        print(f"Learning rate: {lr}, Epochs: {epochs}, Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy_softmax:
            best_accuracy_softmax = val_accuracy
            best_LR_softmax = lr
            best_EPOCHS_softmax = epochs
            best_CL_softmax = CL

print("Best Learning Rate:", best_LR_softmax)
print("Best Epochs:", best_EPOCHS_softmax)
print("Best Accuracy:", best_accuracy_softmax)


#Comparison

print("One-vs-Rest Accuracy:", 0.83)
print("Softmax Accuracy:", best_accuracy_softmax)
print("Difference:", best_accuracy_softmax - 0.83)


#Decision Boundaries

plot_decision_regions(X_train_scaled, t_multi_train, best_CL_softmax)
plt.show()


#Conclusion

#The validation accuracy is 0.848
#The best learning rate is 0.1
#The best number of epochs is 500
#The best accuracy is 84.8%
#One-vs-Rest accuracy is 83%
#Softmax accuracy is 84.8%
#The difference is 1.8 percentage points
#Softmax performs slightly better than One-vs-Rest
#The decision boundaries separate the five classes reasonably well




# First, we define the logistic function and its derivative:
def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_diff(y):
    return y * (1 - y)



class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 6):
        """Intialize the hyperparameters"""
        self.bias = bias
        # Dimensionality of the hidden layer
        self.dim_hidden = dim_hidden
        
        self.activ = logistic
        
        self.activ_diff = logistic_diff
        
    def forward(self, X):
        """TODO: 
        Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        
        hidden = self.activ(X @ self.weights1)
        hidden_outs = add_bias(hidden, self.bias)
        outputs = hidden_outs @ self.weights2

        return hidden_outs, outputs
    
    def fit(self, X_train, t_train, lr=0.001, epochs = 100):
        """Initialize the weights. Train *epochs* many epochs.
        
        X_train is a NxM matrix, N data points, M features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        lr is the learning rate
        """
        self.lr = lr
        
        # Turn t_train into a column vector, a N*1 matrix:
        T_train = t_train.reshape(-1,1)
            
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Initialize the weights
        self.weights1 = (np.random.rand(
            dim_in + 1, 
            self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(
            self.dim_hidden+1, 
            dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)
        
        for e in range(epochs):
            # One epoch
            # The forward step:
            hidden_outs, outputs = self.forward(X_train_bias)
            # The delta term on the output node:
            out_deltas = (outputs - T_train)
            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The deltas at the input to the hidden layer:
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
                                self.activ_diff(hidden_outs[:, 1:]))  

            # Update the weights:
            self.weights2 -= self.lr * hidden_outs.T @ out_deltas
            self.weights1 -= self.lr * X_train_bias.T @ hiddenact_deltas 
            
    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5)

#Training MLP

MLP = MLPBinaryLinRegClass(bias=-1, dim_hidden=6)
MLP.fit(X_train_scaled, t2_train, lr=0.0001, epochs=1000)

print("Training Accuracy:", accuracy(MLP.predict(X_train_scaled), t2_train))
print("Validation Accuracy:", accuracy(MLP.predict(X_val_scaled), t2_val))


#Decision Boundaries

plot_decision_regions(X_train_scaled, t2_train, MLP)
plt.show()

#Conclusion

#The training accuracy is 0.735
#The validation accuracy is 0.792
#The MLP performs better than the previous binary logistic regression model
#The decision boundary is still approximately linear

#Hyperparameter Tuning

learning_rates = [0.00005, 0.0001, 0.0002]
epochs_list = [500, 1000, 2000]

best_accuracy = 0
best_LR = None
best_EPOCHS = None
best_MLP = None

for lr in learning_rates:
    for epochs in epochs_list:
        np.random.seed(2024)
        MLP = MLPBinaryLinRegClass(bias=-1, dim_hidden=6)
        MLP.fit(X_train_scaled, t2_train, lr=lr, epochs=epochs)

        val_accuracy = accuracy(MLP.predict(X_val_scaled), t2_val)
        print(f"Learning rate: {lr}, Epochs: {epochs}, Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_LR = lr
            best_EPOCHS = epochs
            best_MLP = MLP

print("Best Learning Rate:", best_LR)
print("Best Epochs:", best_EPOCHS)
print("Best Accuracy:", best_accuracy)

#Conclusion

#Best Learning Rate: 0.0002
#Best Epochs: 2000
#Best Accuracy: 0.8
#Best Training Accuracy: 0.789
#Best Validation Accuracy: 0.8

#The MLP performs better than the previous binary logistic regression model
#The validation accuracy improved from 0.762 to 0.8
#More epochs generally improved the accuracy in the tested settings
print("Best Training Accuracy:", accuracy(best_MLP.predict(X_train_scaled), t2_train))
print("Best Validation Accuracy:", accuracy(best_MLP.predict(X_val_scaled), t2_val))

plot_decision_regions(X_train_scaled, t2_train, best_MLP)
plt.show()




#My own code 

#My own code

#Part a

class ImprovedMLPBinaryClass(MLPBinaryLinRegClass):
    def __init__(self, bias=-1, dim_hidden=6):
        super().__init__(bias=bias, dim_hidden=dim_hidden)

    def forward(self, X):
        hidden = self.activ(X @ self.weights1)
        hidden_outs = add_bias(hidden, self.bias)
        outputs = logistic(hidden_outs @ self.weights2)
        return hidden_outs, outputs

    def fit(self, X_train, t_train, lr=0.1, epochs=500):
        N, D = X_train.shape
        T = t_train.reshape(-1, 1)

        #Initialize the weights
        self.weights1 = (np.random.rand(D + 1, self.dim_hidden) * 2 - 1) / np.sqrt(D)
        self.weights2 = (np.random.rand(self.dim_hidden + 1, 1) * 2 - 1) / np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)

        for epoch in range(epochs):
            #Forward
            hidden_outs, outputs = self.forward(X_train_bias)

            #Backpropagation
            out_deltas = outputs - T
            hiddenact_deltas = (out_deltas @ self.weights2[1:, :].T) * self.activ_diff(hidden_outs[:, 1:])

            #Gradient descent
            self.weights2 -= lr * hidden_outs.T @ out_deltas / N
            self.weights1 -= lr * X_train_bias.T @ hiddenact_deltas / N

        return self

    def predict_probability(self, X):
        outputs = self.forward(add_bias(X, self.bias))[1]
        return outputs[:, 0]

    def predict(self, X):
        return (self.predict_probability(X) >= 0.5).astype(int)


#Test Part a

np.random.seed(2024)

MLP_new = ImprovedMLPBinaryClass(bias=-1, dim_hidden=6)
MLP_new.fit(X_train_scaled, t2_train, lr=0.1, epochs=500)

print("Probabilities:", MLP_new.predict_probability(X_val_scaled[:5]))
print("Predictions:", MLP_new.predict(X_val_scaled[:5]))
print("Validation Accuracy:", accuracy(MLP_new.predict(X_val_scaled), t2_val))


#Part b

class ImprovedMLPBinaryClassB(ImprovedMLPBinaryClass):
    def fit(self, X_train, t_train, lr=0.1, epochs=500):
        N, D = X_train.shape
        T = t_train.reshape(-1, 1)

        #Initialize the weights
        self.weights1 = (np.random.rand(D + 1, self.dim_hidden) * 2 - 1) / np.sqrt(D)
        self.weights2 = (np.random.rand(self.dim_hidden + 1, 1) * 2 - 1) / np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)

        self.history = []
        self.accuracy_history = []

        for epoch in range(epochs):
            #Forward
            hidden_outs, outputs = self.forward(X_train_bias)

            #Backpropagation
            out_deltas = outputs - T
            hiddenact_deltas = (out_deltas @ self.weights2[1:, :].T) * self.activ_diff(hidden_outs[:, 1:])

            #Gradient descent
            self.weights2 -= lr * hidden_outs.T @ out_deltas / N
            self.weights1 -= lr * X_train_bias.T @ hiddenact_deltas / N

            #Training Loss and Accuracy
            p_train = self.predict_probability(X_train)
            train_loss = binary_cross_entropy(t_train, p_train)
            train_accuracy = np.mean((p_train >= 0.5) == t_train)

            self.history.append(train_loss)
            self.accuracy_history.append(train_accuracy)

        return self

#Part c

class ImprovedMLPBinaryClassC(ImprovedMLPBinaryClassB):
    def fit(self, X_train, t_train, lr=0.1, epochs=500, X_val=None, t_val=None):
        if (X_val is None) != (t_val is None):
            raise ValueError("Provide both X_val and t_val")

        N, D = X_train.shape
        T = t_train.reshape(-1, 1)

        #Initialize the weights
        self.weights1 = (np.random.rand(D + 1, self.dim_hidden) * 2 - 1) / np.sqrt(D)
        self.weights2 = (np.random.rand(self.dim_hidden + 1, 1) * 2 - 1) / np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)

        self.history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []

        for epoch in range(epochs):
            #Forward
            hidden_outs, outputs = self.forward(X_train_bias)

            #Backpropagation
            out_deltas = outputs - T
            hiddenact_deltas = (out_deltas @ self.weights2[1:, :].T) * self.activ_diff(hidden_outs[:, 1:])

            #Gradient descent
            self.weights2 -= lr * hidden_outs.T @ out_deltas / N
            self.weights1 -= lr * X_train_bias.T @ hiddenact_deltas / N

            #Training Loss and Accuracy
            p_train = self.predict_probability(X_train)
            train_loss = binary_cross_entropy(t_train, p_train)
            train_accuracy = np.mean((p_train >= 0.5) == t_train)

            self.history.append(train_loss)
            self.accuracy_history.append(train_accuracy)

            #Validation Loss and Accuracy
            if X_val is not None:
                p_val = self.predict_probability(X_val)
                val_loss = binary_cross_entropy(t_val, p_val)
                val_accuracy = np.mean((p_val >= 0.5) == t_val)

                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_accuracy)

        return self

#Test Part c

np.random.seed(2024)

MLP_c = ImprovedMLPBinaryClassC(bias=-1, dim_hidden=6)
MLP_c.fit(X_train_scaled, t2_train, lr=0.1, epochs=500, X_val=X_val_scaled, t_val=t2_val)

print("Training Loss:", MLP_c.history[-1])
print("Training Accuracy:", MLP_c.accuracy_history[-1])

print("Validation Loss:", MLP_c.val_loss_history[-1])
print("Validation Accuracy:", MLP_c.val_accuracy_history[-1])

print("Number of Epochs:", len(MLP_c.val_loss_history))

#Part d

class ImprovedMLPBinaryClassD(ImprovedMLPBinaryClassC):
    def fit(self, X_train, t_train, lr=0.1, epochs=500, X_val=None, t_val=None, tol=0.0001, n_epochs_no_update=5):
        if (X_val is None) != (t_val is None):
            raise ValueError("Provide both X_val and t_val")

        N, D = X_train.shape
        T = t_train.reshape(-1, 1)

        #Initialize the weights
        self.weights1 = (np.random.rand(D + 1, self.dim_hidden) * 2 - 1) / np.sqrt(D)
        self.weights2 = (np.random.rand(self.dim_hidden + 1, 1) * 2 - 1) / np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)

        self.history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []

        #Early Stopping
        best_loss = float("inf")
        no_update = 0
        self.n_epochs_trained = 0

        for epoch in range(epochs):
            #Forward
            hidden_outs, outputs = self.forward(X_train_bias)

            #Backpropagation
            out_deltas = outputs - T
            hiddenact_deltas = (out_deltas @ self.weights2[1:, :].T) * self.activ_diff(hidden_outs[:, 1:])

            #Gradient descent
            self.weights2 -= lr * hidden_outs.T @ out_deltas / N
            self.weights1 -= lr * X_train_bias.T @ hiddenact_deltas / N

            #Training Loss and Accuracy
            p_train = self.predict_probability(X_train)
            train_loss = binary_cross_entropy(t_train, p_train)
            train_accuracy = np.mean((p_train >= 0.5) == t_train)

            self.history.append(train_loss)
            self.accuracy_history.append(train_accuracy)

            #Validation Loss and Accuracy
            if X_val is not None:
                p_val = self.predict_probability(X_val)
                val_loss = binary_cross_entropy(t_val, p_val)
                val_accuracy = np.mean((p_val >= 0.5) == t_val)

                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_accuracy)

            #Early Stopping
            self.n_epochs_trained = epoch + 1
            monitored_loss = val_loss if X_val is not None else train_loss

            if tol is not None:
                if best_loss - monitored_loss > tol:
                    best_loss = monitored_loss
                    no_update = 0
                else:
                    no_update += 1

                if no_update >= n_epochs_no_update:
                    break

        return self

#Test Part d

np.random.seed(2024)

MLP_d = ImprovedMLPBinaryClassD(bias=-1, dim_hidden=6)
MLP_d.fit(X_train_scaled, t2_train, lr=0.1, epochs=1000, X_val=X_val_scaled, t_val=t2_val, tol=0.0001, n_epochs_no_update=5)

print("Epochs trained:", MLP_d.n_epochs_trained)
print("Training Loss:", MLP_d.history[-1])
print("Training Accuracy:", MLP_d.accuracy_history[-1])
print("Validation Loss:", MLP_d.val_loss_history[-1])
print("Validation Accuracy:", MLP_d.val_accuracy_history[-1])


#Part e

learning_rates = [0.05, 0.1, 0.2]
tolerances = [0.001, 0.0001]
hidden_sizes = [6, 12]

datasets = [("Original", X_train, X_val), ("Scaled", X_train_scaled, X_val_scaled)]

best_accuracy = 0
best_LR = None
best_TOL = None
best_HIDDEN = None
best_MLP = None
best_DATA = None
best_epochs = float("inf")

for data_name, X_tr, X_v in datasets:
    for lr in learning_rates:
        for tol in tolerances:
            for hidden in hidden_sizes:
                np.random.seed(2024)

                MLP = ImprovedMLPBinaryClassD(bias=-1, dim_hidden=hidden)
                MLP.fit(X_tr, t2_train, lr=lr, epochs=1000, X_val=X_v, t_val=t2_val, tol=tol, n_epochs_no_update=5)

                val_accuracy = accuracy(MLP.predict(X_v), t2_val)
                print(f"Data: {data_name}, LR: {lr}, Tol: {tol}, Hidden: {hidden}, Epochs: {MLP.n_epochs_trained}, Accuracy: {val_accuracy:.4f}")

                if val_accuracy > best_accuracy or (np.isclose(val_accuracy, best_accuracy) and MLP.n_epochs_trained < best_epochs):
                    best_accuracy = val_accuracy
                    best_LR = lr
                    best_TOL = tol
                    best_HIDDEN = hidden
                    best_MLP = MLP
                    best_DATA = data_name
                    best_epochs = MLP.n_epochs_trained

print("Best Data:", best_DATA)
print("Best Learning Rate:", best_LR)
print("Best Tolerance:", best_TOL)
print("Best Hidden Size:", best_HIDDEN)
print("Best Epochs:", best_epochs)
print("Best Accuracy:", best_accuracy)

#Part f

#Training and Validation Loss
plt.figure(figsize=(8,5))
plt.plot(best_MLP.history, label="Training Loss")
plt.plot(best_MLP.val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

#Training and Validation Accuracy
plt.figure(figsize=(8,5))
plt.plot(best_MLP.accuracy_history, label="Training Accuracy")
plt.plot(best_MLP.val_accuracy_history, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

#Part g

accuracies = []

if best_DATA == "Scaled":
    X_tr = X_train_scaled
    X_v = X_val_scaled
else:
    X_tr = X_train
    X_v = X_val

for run in range(10):
    np.random.seed(2024 + run)

    MLP = ImprovedMLPBinaryClassD(bias=-1, dim_hidden=best_HIDDEN)
    MLP.fit(X_tr, t2_train, lr=best_LR, epochs=1000, X_val=X_v, t_val=t2_val, tol=best_TOL, n_epochs_no_update=5)

    val_accuracy = accuracy(MLP.predict(X_v), t2_val)
    accuracies.append(val_accuracy)

    print(f"Run {run + 1}, Accuracy: {val_accuracy:.4f}")

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("Mean Accuracy:", mean_accuracy)
print("Standard Deviation:", std_accuracy)

#Conclusion

#The best validation accuracy is 0.852
#The best learning rate is 0.2
#The best tolerance is 0.0001
#The best hidden size is 6
#The best number of epochs is 1000

#The training and validation losses decrease as the model learns
#The training and validation accuracies generally increase
#There is no clear evidence of overfitting

#The original data performs better than the scaled data in our experiments
#Early stopping can reduce the number of training epochs

#After 10 runs, the mean validation accuracy is 0.8446
#The standard deviation is 0.00594
#The results vary slightly due to random weight initialization



# Your code here

#My own code

#Multi-class Neural Network

class MLPMultiClass(MLPBinaryLinRegClass):
    def __init__(self, bias=-1, dim_hidden=6):
        super().__init__(bias=bias, dim_hidden=dim_hidden)
        self.classes = None
        self.history = []

    def forward(self, X):
        hidden = self.activ(X @ self.weights1)
        hidden_outs = add_bias(hidden, self.bias)
        outputs = softmax(hidden_outs @ self.weights2)
        return hidden_outs, outputs

    def fit(self, X_train, t_train, lr=0.1, epochs=1000):
        N, D = X_train.shape
        self.classes = np.unique(t_train)
        n_classes = len(self.classes)

        #One-hot encoding
        T = (t_train[:, None] == self.classes).astype(int)

        #Initialize the weights
        self.weights1 = (np.random.rand(D + 1, self.dim_hidden) * 2 - 1) / np.sqrt(D)
        self.weights2 = (np.random.rand(self.dim_hidden + 1, n_classes) * 2 - 1) / np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)
        self.history = []

        for epoch in range(epochs):
            #Forward
            hidden_outs, outputs = self.forward(X_train_bias)

            #Backpropagation
            out_deltas = outputs - T
            hiddenact_deltas = (out_deltas @ self.weights2[1:, :].T) * self.activ_diff(hidden_outs[:, 1:])

            #Gradient descent
            self.weights2 -= lr * hidden_outs.T @ out_deltas / N
            self.weights1 -= lr * X_train_bias.T @ hiddenact_deltas / N

            #Training Loss
            probabilities = self.forward(X_train_bias)[1]
            loss = multiclass_cross_entropy(T, probabilities) / N
            self.history.append(loss)

        return self

    def predict_probability(self, X):
        outputs = self.forward(add_bias(X, self.bias))[1]
        return outputs

    def predict(self, X):
        probabilities = self.predict_probability(X)
        predictions = np.argmax(probabilities, axis=1)
        return self.classes[predictions]



#Test Multi-class MLP

np.random.seed(2024)

MLP_multi = MLPMultiClass(bias=-1, dim_hidden=6)
MLP_multi.fit(X_train_scaled, t_multi_train, lr=0.1, epochs=1000)

print("Training Accuracy:", accuracy(MLP_multi.predict(X_train_scaled), t_multi_train))
print("Validation Accuracy:", accuracy(MLP_multi.predict(X_val_scaled), t_multi_val))
print("First 5 Predictions:", MLP_multi.predict(X_val_scaled[:5]))
print("First 5 Probabilities:", MLP_multi.predict_probability(X_val_scaled[:5]))



#Check probabilities

probabilities = MLP_multi.predict_probability(X_val_scaled[:5])

print("Probability sums:", np.sum(probabilities, axis=1))
print("Weights1 Shape:", MLP_multi.weights1.shape)
print("Weights2 Shape:", MLP_multi.weights2.shape)

#Different learning rates

learning_rates = [0.05, 0.1, 0.2]

for lr in learning_rates:
    np.random.seed(2024)
    MLP = MLPMultiClass(bias=-1, dim_hidden=6)
    MLP.fit(X_train_scaled, t_multi_train, lr=lr, epochs=1000)

    val_accuracy = accuracy(MLP.predict(X_val_scaled), t_multi_val)
    print(f"Learning Rate: {lr}, Accuracy: {val_accuracy:.4f}")


#Different Epochs

epochs_list = [500, 1000, 1500]

for epochs in epochs_list:
    np.random.seed(2024)
    MLP = MLPMultiClass(bias=-1, dim_hidden=6)
    MLP.fit(X_train_scaled, t_multi_train, lr=0.1, epochs=epochs)

    val_accuracy = accuracy(MLP.predict(X_val_scaled), t_multi_val)
    print(f"Epochs: {epochs}, Accuracy: {val_accuracy:.4f}")


#Different Hidden sizes

hidden_sizes = [6, 12, 20]

for hidden in hidden_sizes:
    np.random.seed(2024)
    MLP = MLPMultiClass(bias=-1, dim_hidden=hidden)
    MLP.fit(X_train_scaled, t_multi_train, lr=0.1, epochs=1000)

    val_accuracy = accuracy(MLP.predict(X_val_scaled), t_multi_val)
    print(f"Hidden Size: {hidden}, Accuracy: {val_accuracy:.4f}")


#Find the best Validation Accuracy

best_accuracy = 0
best_LR = None
best_EPOCHS = None
best_HIDDEN = None
best_MLP = None

for lr in learning_rates:
    for epochs in epochs_list:
        for hidden in hidden_sizes:
            np.random.seed(2024)
            MLP = MLPMultiClass(bias=-1, dim_hidden=hidden)
            MLP.fit(X_train_scaled, t_multi_train, lr=lr, epochs=epochs)

            val_accuracy = accuracy(MLP.predict(X_val_scaled), t_multi_val)
            print(f"LR: {lr}, Epochs: {epochs}, Hidden: {hidden}, Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_LR = lr
                best_EPOCHS = epochs
                best_HIDDEN = hidden
                best_MLP = MLP

print("Best Learning Rate:", best_LR)
print("Best Epochs:", best_EPOCHS)
print("Best Hidden Size:", best_HIDDEN)
print("Best Validation Accuracy:", best_accuracy)


#Plot the best Decision Boundaries

plot_decision_regions(X_train_scaled, t_multi_train, best_MLP)
plt.show()



print("Best Learning Rate:", best_LR)
print("Best Epochs:", best_EPOCHS)
print("Best Hidden Size:", best_HIDDEN)
print("Best Training Accuracy:", accuracy(best_MLP.predict(X_train_scaled), t_multi_train))
print("Best Validation Accuracy:", best_accuracy)
print("Previous Softmax Accuracy:", 0.848)
print("Accuracy Difference:", round(best_accuracy - 0.848, 4))


#Conclusion

# Your code here

#My own code

#Binary Task

#Linear Regression Classifier
CL_linear = NumpyLinRegClass()
CL_linear.fit(X_train_scaled, t2_train, lr=0.05, epochs=500)

#Logistic Regression Classifier
CL_logistic = NumpyLogRegClassExtended()
CL_logistic.fit(X_train_scaled, t2_train, learning_rate=0.01, epochs=1000, X_val=X_val_scaled, t_val=t2_val, tol=0.001)

#Multi-layer Neural Network
np.random.seed(2024)
CL_mlp = ImprovedMLPBinaryClassD(bias=-1, dim_hidden=6)
CL_mlp.fit(X_train, t2_train, lr=0.2, epochs=1000, X_val=X_val, t_val=t2_val, tol=0.0001)

#Accuracy Comparison
models = [("Linear Regression", CL_linear, True), ("Logistic Regression", CL_logistic, True), ("MLP", CL_mlp, False)]

print(f"{'Model':<25} {'Train':<10} {'Validation':<12} {'Test':<10}")

for name, model, scaled in models:
    X_tr = X_train_scaled if scaled else X_train
    X_v = X_val_scaled if scaled else X_val
    X_te = scaler.transform(X_test) if scaled else X_test

    train_acc = accuracy(model.predict(X_tr), t2_train)
    val_acc = accuracy(model.predict(X_v), t2_val)
    test_acc = accuracy(model.predict(X_te), t2_test)

    print(f"{name:<25} {train_acc:<10.4f} {val_acc:<12.4f} {test_acc:<10.4f}")



#Precision and Recall for Class 1

for name, model, scaled in models:
    X_te = scaler.transform(X_test) if scaled else X_test
    predictions = model.predict(X_te)

    TP = np.sum((predictions == 1) & (t2_test == 1))
    FP = np.sum((predictions == 1) & (t2_test == 0))
    FN = np.sum((predictions == 0) & (t2_test == 1))

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    print(f"{name}: Precision = {precision:.4f}, Recall = {recall:.4f}")



#Conclusion

#The MLP has the best test accuracy of 0.822
#Logistic Regression achieves 0.724
#Linear Regression achieves 0.720

#The test accuracy is lower than the validation accuracy for all three models
#The MLP performs better than the two linear classifiers
#The MLP also has the highest precision and recall for class 1

#The MLP precision is 0.7368
#The MLP recall is 0.8528
#There is no clear evidence of severe overfitting




# Your code here



#My own code

#Multi-class Task

#One-vs-Rest Logistic Regression
CL_ovr_final = OneVsRest()
CL_ovr_final.fit(X_train_scaled, t_multi_train, learning_rate=0.1, epochs=500)

#Multinomial Logistic Regression
CL_softmax_final = NumpySoftmaxClass()
CL_softmax_final.fit(X_train_scaled, t_multi_train, learning_rate=0.1, epochs=500)

#Multi-class Neural Network
np.random.seed(2024)
CL_mlp_final = MLPMultiClass(bias=-1, dim_hidden=20)
CL_mlp_final.fit(X_train_scaled, t_multi_train, lr=0.2, epochs=1500)

#Scaling the Test Set
X_test_scaled = scaler.transform(X_test)

#Accuracy Comparison
models_multi = [("One-vs-Rest", CL_ovr_final), ("Softmax", CL_softmax_final), ("Multi-class MLP", CL_mlp_final)]

print(f"{'Model':<25} {'Train':<10} {'Validation':<12} {'Test':<10}")

for name, model in models_multi:
    train_acc = accuracy(model.predict(X_train_scaled), t_multi_train)
    val_acc = accuracy(model.predict(X_val_scaled), t_multi_val)
    test_acc = accuracy(model.predict(X_test_scaled), t_multi_test)

    print(f"{name:<25} {train_acc:<10.4f} {val_acc:<12.4f} {test_acc:<10.4f}")


#Conclusion
#The MLP has the best test accuracy of 0.838
#Softmax achieves a test accuracy of 0.792
#One-vs-Rest achieves a test accuracy of 0.776
#The MLP performs better than the two logistic regression classifiers
#The test accuracy is lower than the validation accuracy for all three models
#There is no clear evidence of severe overfitting
#The MLP can learn non-linear relationships between the features
#The results show that the MLP performs best on this dataset


"""
Accuracy on the validation set: 0.58
Learning rate: 0.001, Epochs: 10, Accuracy: 0.6040
Learning rate: 0.001, Epochs: 100, Accuracy: 0.4880
Learning rate: 0.001, Epochs: 500, Accuracy: 0.5080
Learning rate: 0.01, Epochs: 10, Accuracy: 0.4700
Learning rate: 0.01, Epochs: 100, Accuracy: 0.5660
Learning rate: 0.01, Epochs: 500, Accuracy: 0.6960
Learning rate: 0.05, Epochs: 10, Accuracy: 0.5080
Learning rate: 0.05, Epochs: 100, Accuracy: 0.6960
Learning rate: 0.05, Epochs: 500, Accuracy: 0.7600
Best learning rate: 0.05
Best epochs: 500
Best accuracy: 0.76
Mean: [ 9.17044218e-17 -1.87072580e-16]
Std: [1. 1.]
Shape: (1000, 2)
Learning rate: 0.001, Epochs: 10, Accuracy: 0.6040
Learning rate: 0.001, Epochs: 100, Accuracy: 0.6040
Learning rate: 0.001, Epochs: 500, Accuracy: 0.6040
Learning rate: 0.01, Epochs: 10, Accuracy: 0.6040
Learning rate: 0.01, Epochs: 100, Accuracy: 0.6000
Learning rate: 0.01, Epochs: 500, Accuracy: 0.7540
Learning rate: 0.05, Epochs: 10, Accuracy: 0.6040
Learning rate: 0.05, Epochs: 100, Accuracy: 0.7540
Learning rate: 0.05, Epochs: 500, Accuracy: 0.7600
Best learning rate: 0.05
Best epochs: 500
Best accuracy: 0.76
Validation Accuracy: 0.76
Probabilities: [0.17055791 0.54043214 0.48459407 0.21852645 0.58091734]
First Training Loss: 0.6931471805599454
Final Training Loss: 0.527312910714661
First Training Accuracy: 0.405
Final Training Accuracy: 0.719
Training Loss: 0.5273119856850044
Training Accuracy: 0.719
Validation Loss: 0.49120205462479816
Validation Accuracy: 0.76
Epochs trained: 401
Validation Accuracy: 0.76
Validation Loss: 0.4925771173043021
Original, LR: 0.01, Tolerance: 0.001, Epochs: 251, Accuracy: 0.7340
Original, LR: 0.01, Tolerance: 0.0001, Epochs: 1000, Accuracy: 0.7540
Original, LR: 0.01, Tolerance: 1e-05, Epochs: 1000, Accuracy: 0.7540
Original, LR: 0.05, Tolerance: 0.001, Epochs: 146, Accuracy: 0.7540
Original, LR: 0.05, Tolerance: 0.0001, Epochs: 415, Accuracy: 0.7560
Original, LR: 0.05, Tolerance: 1e-05, Epochs: 913, Accuracy: 0.7600
Original, LR: 0.1, Tolerance: 0.001, Epochs: 104, Accuracy: 0.7560
Original, LR: 0.1, Tolerance: 0.0001, Epochs: 272, Accuracy: 0.7560
Original, LR: 0.1, Tolerance: 1e-05, Epochs: 559, Accuracy: 0.7600
Scaled, LR: 0.01, Tolerance: 0.001, Epochs: 291, Accuracy: 0.7620
Scaled, LR: 0.01, Tolerance: 0.0001, Epochs: 1000, Accuracy: 0.7620
Scaled, LR: 0.01, Tolerance: 1e-05, Epochs: 1000, Accuracy: 0.7620
Scaled, LR: 0.05, Tolerance: 0.001, Epochs: 199, Accuracy: 0.7620
Scaled, LR: 0.05, Tolerance: 0.0001, Epochs: 612, Accuracy: 0.7620
Scaled, LR: 0.05, Tolerance: 1e-05, Epochs: 1000, Accuracy: 0.7600
Scaled, LR: 0.1, Tolerance: 0.001, Epochs: 148, Accuracy: 0.7620
Scaled, LR: 0.1, Tolerance: 0.0001, Epochs: 401, Accuracy: 0.7600
Scaled, LR: 0.1, Tolerance: 1e-05, Epochs: 766, Accuracy: 0.7600
Best configuration: ('Scaled', 0.01, 0.001)
Best accuracy: 0.762
Best epochs: 291
Validation Accuracy: 0.83
First 5 predictions: [0 4 3 2 3]
First 5 true labels: [0 4 3 2 3]
Learning rate: 0.01, Epochs: 100, Accuracy: 0.7420
Learning rate: 0.01, Epochs: 300, Accuracy: 0.7460
Learning rate: 0.01, Epochs: 500, Accuracy: 0.7560
Learning rate: 0.05, Epochs: 100, Accuracy: 0.7560
Learning rate: 0.05, Epochs: 300, Accuracy: 0.7860
Learning rate: 0.05, Epochs: 500, Accuracy: 0.8180
Learning rate: 0.1, Epochs: 100, Accuracy: 0.7800
Learning rate: 0.1, Epochs: 300, Accuracy: 0.8280
Learning rate: 0.1, Epochs: 500, Accuracy: 0.8300
Best Learning Rate: 0.1
Best Epochs: 500
Best Accuracy: 0.83
Validation Accuracy: 0.848
First 5 predictions: [0 4 3 2 3]
First 5 probabilities: [[0.54866974 0.27590503 0.00284671 0.15848777 0.01409075]
 [0.00723356 0.14415057 0.2480979  0.09562932 0.50488865]
 [0.32831797 0.14580826 0.00445669 0.46467996 0.05673712]
 [0.00188341 0.19614165 0.57332239 0.01239364 0.2162589 ]
 [0.21979255 0.12822392 0.00704026 0.54745248 0.0974908 ]]
Learning rate: 0.01, Epochs: 100, Accuracy: 0.7420
Learning rate: 0.01, Epochs: 300, Accuracy: 0.7600
Learning rate: 0.01, Epochs: 500, Accuracy: 0.7800
Learning rate: 0.05, Epochs: 100, Accuracy: 0.7800
Learning rate: 0.05, Epochs: 300, Accuracy: 0.8140
Learning rate: 0.05, Epochs: 500, Accuracy: 0.8160
Learning rate: 0.1, Epochs: 100, Accuracy: 0.8040
Learning rate: 0.1, Epochs: 300, Accuracy: 0.8200
Learning rate: 0.1, Epochs: 500, Accuracy: 0.8480
Best Learning Rate: 0.1
Best Epochs: 500
Best Accuracy: 0.848
One-vs-Rest Accuracy: 0.83
Softmax Accuracy: 0.848
Difference: 0.018000000000000016
Training Accuracy: 0.734
Validation Accuracy: 0.784
Learning rate: 5e-05, Epochs: 500, Accuracy: 0.7740
Learning rate: 5e-05, Epochs: 1000, Accuracy: 0.7940
Learning rate: 5e-05, Epochs: 2000, Accuracy: 0.7900
Learning rate: 0.0001, Epochs: 500, Accuracy: 0.7940
Learning rate: 0.0001, Epochs: 1000, Accuracy: 0.7900
Learning rate: 0.0001, Epochs: 2000, Accuracy: 0.7900
Learning rate: 0.0002, Epochs: 500, Accuracy: 0.7900
Learning rate: 0.0002, Epochs: 1000, Accuracy: 0.7900
Learning rate: 0.0002, Epochs: 2000, Accuracy: 0.8000
Best Learning Rate: 0.0002
Best Epochs: 2000
Best Accuracy: 0.8
Best Training Accuracy: 0.789
Best Validation Accuracy: 0.8
Probabilities: [0.21454808 0.53221128 0.48509079 0.2597336  0.55069686]
Predictions: [0 1 0 0 1]
Validation Accuracy: 0.762
Training Loss: 0.5217765842566289
Training Accuracy: 0.72
Validation Loss: 0.5065818081109563
Validation Accuracy: 0.762
Number of Epochs: 500
Epochs trained: 1000
Training Loss: 0.48883470907904975
Training Accuracy: 0.737
Validation Loss: 0.4670857848070034
Validation Accuracy: 0.788
Data: Original, LR: 0.05, Tol: 0.001, Hidden: 6, Epochs: 575, Accuracy: 0.7440
Data: Original, LR: 0.05, Tol: 0.001, Hidden: 12, Epochs: 43, Accuracy: 0.6040
Data: Original, LR: 0.05, Tol: 0.0001, Hidden: 6, Epochs: 1000, Accuracy: 0.8060
Data: Original, LR: 0.05, Tol: 0.0001, Hidden: 12, Epochs: 1000, Accuracy: 0.7720
Data: Original, LR: 0.1, Tol: 0.001, Hidden: 6, Epochs: 497, Accuracy: 0.8060
Data: Original, LR: 0.1, Tol: 0.001, Hidden: 12, Epochs: 504, Accuracy: 0.7760
Data: Original, LR: 0.1, Tol: 0.0001, Hidden: 6, Epochs: 1000, Accuracy: 0.8300
Data: Original, LR: 0.1, Tol: 0.0001, Hidden: 12, Epochs: 1000, Accuracy: 0.8300
Data: Original, LR: 0.2, Tol: 0.001, Hidden: 6, Epochs: 400, Accuracy: 0.8260
Data: Original, LR: 0.2, Tol: 0.001, Hidden: 12, Epochs: 429, Accuracy: 0.8200
Data: Original, LR: 0.2, Tol: 0.0001, Hidden: 6, Epochs: 1000, Accuracy: 0.8520
Data: Original, LR: 0.2, Tol: 0.0001, Hidden: 12, Epochs: 1000, Accuracy: 0.8480
Data: Scaled, LR: 0.05, Tol: 0.001, Hidden: 6, Epochs: 16, Accuracy: 0.6040
Data: Scaled, LR: 0.05, Tol: 0.001, Hidden: 12, Epochs: 101, Accuracy: 0.6040
Data: Scaled, LR: 0.05, Tol: 0.0001, Hidden: 6, Epochs: 1000, Accuracy: 0.7620
Data: Scaled, LR: 0.05, Tol: 0.0001, Hidden: 12, Epochs: 1000, Accuracy: 0.7740
Data: Scaled, LR: 0.1, Tol: 0.001, Hidden: 6, Epochs: 481, Accuracy: 0.7580
Data: Scaled, LR: 0.1, Tol: 0.001, Hidden: 12, Epochs: 467, Accuracy: 0.7700
Data: Scaled, LR: 0.1, Tol: 0.0001, Hidden: 6, Epochs: 1000, Accuracy: 0.7880
Data: Scaled, LR: 0.1, Tol: 0.0001, Hidden: 12, Epochs: 1000, Accuracy: 0.7900
Data: Scaled, LR: 0.2, Tol: 0.001, Hidden: 6, Epochs: 323, Accuracy: 0.7840
Data: Scaled, LR: 0.2, Tol: 0.001, Hidden: 12, Epochs: 310, Accuracy: 0.7740
Data: Scaled, LR: 0.2, Tol: 0.0001, Hidden: 6, Epochs: 1000, Accuracy: 0.7900
Data: Scaled, LR: 0.2, Tol: 0.0001, Hidden: 12, Epochs: 1000, Accuracy: 0.8020
Best Data: Original
Best Learning Rate: 0.2
Best Tolerance: 0.0001
Best Hidden Size: 6
Best Epochs: 1000
Best Accuracy: 0.852
Run 1, Accuracy: 0.8520
Run 2, Accuracy: 0.8440
Run 3, Accuracy: 0.8420
Run 4, Accuracy: 0.8540
Run 5, Accuracy: 0.8420
Run 6, Accuracy: 0.8480
Run 7, Accuracy: 0.8400
Run 8, Accuracy: 0.8340
Run 9, Accuracy: 0.8500
Run 10, Accuracy: 0.8400
Mean Accuracy: 0.8446
Standard Deviation: 0.005936328831862337
Training Accuracy: 0.798
Validation Accuracy: 0.834
First 5 Predictions: [0 4 3 2 3]
First 5 Probabilities: [[0.59546348 0.22544743 0.00179843 0.16942413 0.00786653]
 [0.00288081 0.12094335 0.24390197 0.06477142 0.56750245]
 [0.32216382 0.14126642 0.00273485 0.4977722  0.03606271]
 [0.00078732 0.19480552 0.59275578 0.00861173 0.20303965]
 [0.20223995 0.13907341 0.00440026 0.58972754 0.06455883]]
Probability sums: [1. 1. 1. 1. 1.]
Weights1 Shape: (3, 6)
Weights2 Shape: (7, 5)
Learning Rate: 0.05, Accuracy: 0.8200
Learning Rate: 0.1, Accuracy: 0.8340
Learning Rate: 0.2, Accuracy: 0.8580
Epochs: 500, Accuracy: 0.8200
Epochs: 1000, Accuracy: 0.8340
Epochs: 1500, Accuracy: 0.8460
Hidden Size: 6, Accuracy: 0.8340
Hidden Size: 12, Accuracy: 0.8640
Hidden Size: 20, Accuracy: 0.8720
LR: 0.05, Epochs: 500, Hidden: 6, Accuracy: 0.6640
LR: 0.05, Epochs: 500, Hidden: 12, Accuracy: 0.7900
LR: 0.05, Epochs: 500, Hidden: 20, Accuracy: 0.8240
LR: 0.05, Epochs: 1000, Hidden: 6, Accuracy: 0.8200
LR: 0.05, Epochs: 1000, Hidden: 12, Accuracy: 0.8680
LR: 0.05, Epochs: 1000, Hidden: 20, Accuracy: 0.8520
LR: 0.05, Epochs: 1500, Hidden: 6, Accuracy: 0.8300
LR: 0.05, Epochs: 1500, Hidden: 12, Accuracy: 0.8640
LR: 0.05, Epochs: 1500, Hidden: 20, Accuracy: 0.8680
LR: 0.1, Epochs: 500, Hidden: 6, Accuracy: 0.8200
LR: 0.1, Epochs: 500, Hidden: 12, Accuracy: 0.8680
LR: 0.1, Epochs: 500, Hidden: 20, Accuracy: 0.8520
LR: 0.1, Epochs: 1000, Hidden: 6, Accuracy: 0.8340
LR: 0.1, Epochs: 1000, Hidden: 12, Accuracy: 0.8640
LR: 0.1, Epochs: 1000, Hidden: 20, Accuracy: 0.8720
LR: 0.1, Epochs: 1500, Hidden: 6, Accuracy: 0.8460
LR: 0.1, Epochs: 1500, Hidden: 12, Accuracy: 0.8700
LR: 0.1, Epochs: 1500, Hidden: 20, Accuracy: 0.8740
LR: 0.2, Epochs: 500, Hidden: 6, Accuracy: 0.8340
LR: 0.2, Epochs: 500, Hidden: 12, Accuracy: 0.8640
LR: 0.2, Epochs: 500, Hidden: 20, Accuracy: 0.8720
LR: 0.2, Epochs: 1000, Hidden: 6, Accuracy: 0.8580
LR: 0.2, Epochs: 1000, Hidden: 12, Accuracy: 0.8740
LR: 0.2, Epochs: 1000, Hidden: 20, Accuracy: 0.8760
LR: 0.2, Epochs: 1500, Hidden: 6, Accuracy: 0.8640
LR: 0.2, Epochs: 1500, Hidden: 12, Accuracy: 0.8720
LR: 0.2, Epochs: 1500, Hidden: 20, Accuracy: 0.8820
Best Learning Rate: 0.2
Best Epochs: 1500
Best Hidden Size: 20
Best Validation Accuracy: 0.882
Best Learning Rate: 0.2
Best Epochs: 1500
Best Hidden Size: 20
Best Training Accuracy: 0.843
Best Validation Accuracy: 0.882
Previous Softmax Accuracy: 0.848
Accuracy Difference: 0.034
Model                     Train      Validation   Test      
Linear Regression         0.7210     0.7600       0.7200    
Logistic Regression       0.7190     0.7620       0.7240    
MLP                       0.8320     0.8520       0.8220    
Linear Regression: Precision = 0.6492, Recall = 0.6294
Logistic Regression: Precision = 0.6561, Recall = 0.6294
MLP: Precision = 0.7368, Recall = 0.8528
Model                     Train      Validation   Test      
One-vs-Rest               0.7790     0.8300       0.7760    
Softmax                   0.7960     0.8480       0.7920    
Multi-class MLP           0.8430     0.8820       0.8380   
"""