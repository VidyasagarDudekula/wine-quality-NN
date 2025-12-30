import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data():
    df = pd.read_csv('WineQT.csv')
    X = df.drop(['quality', 'Id'], axis=1).values
    y = df['quality'].values
    y = y - 3
    print(y.min(), y.max())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data()

print(f"Sample y data:- {y_train[0:5]}")

model = nn.Sequential(
    nn.Linear(11, 20),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(20, 15),
    nn.ReLU(),
    nn.Linear(15, 6),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

history = {'avg_loss': [], 'val_loss': []}
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
for epoch in range(150):
    loss_list = []
    for x_batch, y_batch in dataloader:
        model.train()
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    history['avg_loss'].append(sum(loss_list)/len(loss_list))
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        eval_loss = criterion(predictions, y_test).item()
        history['val_loss'].append(eval_loss)
    if (epoch+1)%10==0:
        print(f"Epoch:- {epoch}, train loss:- {history['avg_loss'][-1]}, eval loss:- {eval_loss}")


print("\n\n")
print("Total History:-")
print(history)