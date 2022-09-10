import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from parameters import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(URL_EMBEDDING, header=0)
df = df.drop(df.columns[0], axis=1)
df.columns = ["EF1", "EF2", "ES1", "ES2", "Metric"]

##########################
#####Create Input and Output Data
##########################

X = df.iloc[:, :-1]
y = df["Metric"].iloc[:]

##########################
#####Train — Validation — Test
##########################

# Train - Test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69
)
# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, random_state=21
)

##########################
#####Normalize Input
##########################

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)


##########################
#####Convert Output Variable to Float
##########################

y_train, y_test, y_val = (
    y_train.astype(float),
    y_test.astype(float),
    y_val.astype(float),
)


##########################
#####       Neural Network
#####       Initialize Dataset
##########################


class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_dataset = RegressionDataset(
    torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
)
val_dataset = RegressionDataset(
    torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
)
test_dataset = RegressionDataset(
    torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
)


##########################
#####    Model Params
##########################

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_FEATURES = len(X.columns)


##########################
#####   Initialize Dataloader
##########################


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


##########################
#####   Define Neural Network Architecture
##########################


class Net(nn.Module):
    def __init__(self, input):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 64)
        self.l4 = nn.Linear(64, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 32)
        self.l7 = nn.Linear(32, 16)
        self.l8 = nn.Linear(16, 4)
        self.oupt = nn.Linear(4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        x = self.relu(self.l5(x))
        x = self.relu(self.l6(x))
        x = self.relu(self.l7(x))
        x = self.relu(self.l8(x))
        x = self.oupt(x)
        return x

    def predict(self, test_x):
        x = self.relu(self.l1(test_x))
        x = self.relu(self.l2(test_x))
        x = self.relu(self.l3(test_x))
        x = self.relu(self.l4(test_x))
        x = self.relu(self.l5(test_x))
        x = self.relu(self.l6(test_x))
        x = self.relu(self.l7(test_x))
        x = self.relu(self.l8(test_x))
        x = self.oupt(x)
        return x


##########################
#####   Check for GPU
##########################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


n = Net(4)
n.to(device)

print(n)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(n.parameters(), lr=0.001)


##########################
#####  Train Model
##########################

# loss_stats = {"train": [], "val": []}


print("Begin training.")
for e in range(1, EPOCHS + 1):

    # TRAINING
    train_epoch_loss = 0
    n.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(
            device
        )
        optimizer.zero_grad()

        y_train_pred = n(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()

    # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0

        n.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = n(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

            val_epoch_loss += val_loss.item()

    # loss_stats['train'].append(train_epoch_loss/len(train_loader))
    # loss_stats['val'].append(val_epoch_loss/len(val_loader))

    print(
        f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}"
    )

y_pred_list = []
with torch.no_grad():
    n.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = n(X_batch)
        y_pred_list.append(y_test_pred.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

y_pred_list = np.array(y_pred_list)


##########################
#####   Evaluate Model
##########################
wrong = [0] * 4
error_7 = 0
error_5 = 0
error_3 = 0
error_1 = 0


for i in range(len(y_pred_list)):
    if round(y_pred_list[i], 0) != y_test[i]:
        # print("Predicted: ", round(y_pred_list[i],0),"-",y_pred_list[i], "Actual: ", y_test[i])
        wrong[0] += 1

    if round(y_pred_list[i], 1) != y_test[i]:
        # print("Predicted: ", round(y_pred_list[i],1),"-",y_pred_list[i], "Actual: ", y_test[i])
        wrong[1] += 1

    if round(y_pred_list[i], 2) != y_test[i]:
        # print("Predicted: ", round(y_pred_list[i],2),"-",y_pred_list[i], "Actual: ", y_test[i])
        wrong[2] += 1

    if round(y_pred_list[i], 3) != y_test[i]:
        # print("Predicted: ", round(y_pred_list[i],3),"-",y_pred_list[i], "Actual: ", y_test[i])
        wrong[3] += 1
    diff = round(y_pred_list[i], 0) - y_test[i]

    if diff > 1:
        # print(
        #     "Predicted: ",
        #     round(y_pred_list[i], 0),
        #     "-",
        #     y_pred_list[i],
        #     "Actual: ",
        #     y_test[i],
        # )
        error_1 += 1
        if diff > 3:
            error_3 += 1
            if diff > 5:
                error_5 += 1
                if diff > 7:
                    error_7 += 1


print(
    wrong,
    len(y_pred_list),
    error_1,
    error_3,
    error_5,
    error_7,
)
