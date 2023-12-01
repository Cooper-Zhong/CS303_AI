# %%
import numpy as np
import util
from SoftmaxRegression import SoftmaxRegression


classification_train_data = util.load_data("./data/classification_train_data.pkl")
classification_train_label = util.load_data("./data/classification_train_label.pkl")
classification_test_data = util.load_data("./data/classification_test_data.pkl")


print("Classification Train Data Shape:", classification_train_data.shape)
print("Classification Train Label Shape:", classification_train_label.shape)
print("Classification Test Data Shape:", classification_test_data.shape)


train_data_index = classification_train_data[:, 0]
train_label_index = classification_train_label[:, 0]
test_data_index = classification_test_data[:, 0]
classification_train_data = classification_train_data[:, 1:]
classification_train_label = classification_train_label[:, 1:].reshape(-1)
classification_test_data = classification_test_data[:, 1:]

classification_train_data.shape, classification_train_label.shape, classification_test_data.shape


train_data_index.shape, train_label_index.shape, test_data_index.shape


# calculate the mean and standard deviation of each column
mean = np.mean(classification_train_data, axis=0)
std_dev = np.std(classification_train_data, axis=0)

# Z-Score normalizes each column
classification_train_data = (classification_train_data - mean) / std_dev
classification_test_data = (classification_test_data - mean) / std_dev

# label one-hot encoding
num_classes =  10 
classification_train_label = np.eye(num_classes)[classification_train_label]
print("train label shape:", classification_train_label.shape)


# ## 4. Dataset Splitting


# divide the data set into training set and validation set
train_ratio = 0.8
seed = 123
(train_data, train_labels), (validation_data, validation_labels) = util.split_train_validation(
    classification_train_data, classification_train_label,
    train_ratio=train_ratio, random_seed=seed
    )

train_data.shape, train_labels.shape, validation_data.shape, validation_labels.shape

# %% [markdown]
# # 5. Model

# %%
linear_model = SoftmaxRegression(
    num_classes=num_classes,
    learning_rate=0.1,
    num_iterations=10000,
    random_seed=seed)

# %% [markdown]
# ## 6. Train 

# %%
train_losses, val_losses, train_accuracies, val_accuracies = linear_model.fit(
    X_train=train_data, y_train=train_labels, 
    X_val=validation_data, y_val=validation_labels
    )

# %%
train_accuracies[-1], val_accuracies[-1]

# %%
util.plot_loss_curves(train_losses=train_losses, val_losses=val_losses)

# %%
util.plot_acc_curves(train_acc=train_accuracies, val_acc=val_accuracies)

# %% [markdown]
# ## 7. Predict

# %%
test_label_predict = linear_model.predict(classification_test_data)

# %%
# merge index and corresponding classification results 
submit_data = np.hstack((
    test_data_index.reshape(-1, 1),
    test_label_predict.reshape(-1, 1)
    ))

# %%
submit_data.shape

# %%
util.save_data('./classification_results.pkl', submit_data)


