import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# ==========================================
# 1. Data and regression basics
# ==========================================


#Define the data
torch.manual_seed(438)
x = torch.rand((50, 1))
y = ( 2 * torch.exp(x) + 50 * torch.cos(x) + 0.05*torch.randn((50, 1)) )


# Linear regression
# Train-test split
n = 100
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=438)
x_train = x_train.numpy()
x_test = x_test.numpy()
y_train = y_train.numpy()
y_test = y_test.numpy()

#Design matrix
X_train = np.vander(x_train.ravel(),10,increasing=True)
X_test = np.vander (x_test.ravel(),10,increasing= True)

#Find the parameter using OLS
thetha_LN = np.linalg.lstsq(X_train,y_train,rcond=0)[0]
print(thetha_LN)

#Prediction
y_train_pred = X_train @ thetha_LN
y_test_pred = X_test @ thetha_LN

# MSE
def MSE(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

MSE_train = MSE(y_train,y_train_pred)
MSE_test = MSE(y_test,y_test_pred)

print("Train MSE", MSE_train)
print("Test MSE", MSE_test)


# R2



# ==========================================
# 2. Bias, variance and model evaluation
# ==========================================

# Bias

# Variance

# Bias-Variance tradeoff

# Bootstrap

# Cross-validation
# K-Fold
# LOOCV


# ==========================================
# 3. Regression methods
# ==========================================

# OLS

# Ridge

# Lasso


# ==========================================
# 4. Linear algebra method
# ==========================================

# SVD


# ==========================================
# 5. Optimization
# ==========================================

# Cost / Loss function

# Gradient

# Learning rate

# Gradient Descent

# Momentum

# AdaGrad

# RMSProp

# Adam