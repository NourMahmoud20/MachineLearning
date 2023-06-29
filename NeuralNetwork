import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import seaborn as sns
import torch.nn as nn
from google.colab import drive
drive.mount('/content/drive')


#read PLCO data March 2022 and NHIS 2019,2020 and 2021
dataPLCO = pd.read_csv('/content/drive/My Drive/PLCO.csv')

data19= pd.read_csv('/content/drive/My Drive/adult19.csv')
data20=pd.read_csv('/content/drive/My Drive/adult20.csv')
data21=pd.read_csv('/content/drive/My Drive/adult21.csv')
dataNHIS = pd.concat([data19, data20, data21])
#Select all rows from all NHIS dataset where sex != 1 (male)
dataNHIS = dataNHIS[dataNHIS.SEX_A != 1]

#Select relevant input features from datasets and merge them

columns_to_drop =[] 

dataPLCO = dataPLCO.drop(columns=columns_to_drop)
my_list = list(dataPLCO)
print (my_list)

dataNH_subset = dataNHIS[['']]

# Rename common columns in dataNHIS to match name in dataPLCO
dataNH_subset = dataNH_subset.rename(columns={''})
#print(dataNH_subset.describe())

# Combine dataPLCO and dataNHIS
merged_data = pd.concat([dataPLCO, dataNH_subset])


merged_data = merged_data.dropna(subset=['ovar_cancer'])


merged_datax=merged_data.drop(columns=['ovar_cancer'])

print(merged_datax.head())
print(merged_datax.describe())
print(merged_datax.info())

# Imputing missing categorical values with mode
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
x_merged_df = imputer.fit_transform(merged_datax)

#Output column
ydata_subset = merged_data['ovar_cancer']
print(ydata_subset.describe())
print(ydata_subset.shape)


#count cancer vs no cancer cases 
num_classes = len(np.unique(ydata_subset))
print("Number of classes:", num_classes)
class_counts = Counter(ydata_subset)
print("Class counts:", class_counts)

#visualize the count
sns.countplot(ydata_subset, label="count classes")
plt.show()


#splitting dataset
#torch.manual_seed(42) # set random seed
X_train, X_test, y_train, y_test = train_test_split(x_merged_df, ydata_subset, test_size=0.2, random_state=42, stratify=ydata_subset)
# Split the training data further into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# convert data to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

X_val=np.array(X_val)
y_val = np.array(y_val)

X_test=np.array(X_test)
y_test = np.array(y_test)


# convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


#Generating the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(99,12)
        self.dropout1 = torch.nn.Dropout(0.7)  # add dropout layer after l1
        self.l2 = torch.nn.Linear(12,1)
        #self.dropout2 = torch.nn.Dropout(0.5)  # add dropout layer after l2
        #self.l3 = torch.nn.Linear(8,1)
        #self.l4 = torch.nn.Linear(4,1)
        
        
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.threshold = 0.5  # set the threshold for binarizing the output
        
    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out1 = self.dropout1(out1)  # apply dropout after l1
        y_pred = self.sigmoid(self.l2(out1))
        #out2 = self.dropout2(out2)  # apply dropout after l2
        #y_pred=self.sigmoid(self.l3(out2))
        #y_pred = self.sigmoid(self.l4(out3))
        #y_pred = self.dropout3(y_pred)  # apply dropout after l3
        # y_pred = self.sigmoid(self.l4(out3))
        y_pred_binary = (y_pred >= self.threshold).float()  # binarize the output based on the threshold
        return y_pred_binary


        #instantiate the model and optimizer for this fold
model = Model()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Create TensorDatasets for training, validation, and testing sets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoader objects for each dataset
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True) #Batch size indicates the number of training examples processed in one iteration.
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


# Define the number of epochs you want to train for
num_epochs = 50
train_losses=[]
val_losses=[]
test_losses=[]


    # Train the model for the specified number of epochs
for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        # Training loop
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
           # weights = labels.eq(1).float() * weight_pos + labels.eq(0).float() * weight_neg
           # loss_fn = nn.MSELoss(reduction='none')
            loss = (loss_fn(outputs.view(-1), labels.float()))
            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate training loss
            train_loss += loss.item()

        for input, labels in val_loader:
            target = model(input)
            loss = loss_fn(target.view(-1),labels)
            # Accumulate validation loss
            val_loss += loss.item()
            # Compute the accuracy of the model on the test set
            print(accuracy_score(target, labels))


        # Record average losses for the epoch
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        # Print epoch and losses
        print(f"Epoch {epoch+1}: Train loss={train_losses[-1]:.4f}, Valid loss={val_losses[-1]:.4f}")



# Set the model to evaluation mode
model.eval()
# Turn off gradients for testing
test_losses=[]
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        
        # Make predictions on the test set
        y_pred = model(inputs)

        # Compute the test loss
        loss = loss_fn(y_pred.view(-1), labels)
        # Accumulate testing loss
        test_loss += loss.item()

        # Compute the accuracy of the model on the test set
        print(accuracy_score(y_pred, y_test))
        
        test_losses.append(test_loss / len(test_loader))

        #Confusion Matrix
        cm = confusion_matrix(y_pred, y_test)
        print(cm)

        # Calculate F1 score
        f1 = f1_score(y_pred, y_test)
        print (f1)

        
        #Classification Report
        report = classification_report(y_test, y_pred)
        print(report)

        print(balanced_accuracy_score(y_test, y_pred))
# Plot losses versus epoch
plt.figure(figsize=(5,5))
plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Testing loss")
plt.plot(val_losses, label="Validation loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

