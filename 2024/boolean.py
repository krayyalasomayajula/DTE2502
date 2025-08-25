#!/usr/bin/env python
# coding: utf-8

# ### Perceptron: Boolean OR implementation

# In[8]:

def sigma(x, w):
    activation = -1.0 * w[-1] #bias
    for i in range(len(x) - 1):
        activation += w[i] * x[i]
    return 1.0 if activation >= 0 else 0.0

def training(data, w0, mu, T):
    w = w0
    for idx in range(T):
        for x in data:
            activation = sigma(x, w)
            error = x[-1] - activation
            w[-1] += -1.0* mu * error

            for i in range(len(x) - 1):
                w[i] += mu * error * x[i]

    return w

# In[9]:
# initialization
dataset = [[0, 0, 0],
           [1, 0, 1],
           [0, 1, 1],
           [1, 1, 1],
          ]
weights = [0.02, -0.03, -1.05]
#training
weights = training(dataset, weights, 0.2, 15)

#inference
for sample in dataset:
    a = sigma(sample, weights)
    print(f"Target: {sample[-1]}, prediction: {a}")

print(f"Final weights: {weights}")
# %%
