import torch
import torch.nn as nn
import matplotlib.pyplot as plt

filename = "data_miasta.csv"

file = open(filename, "r")
data = file.read().split("\n")[:-1]
file.close()

INPUT_N = 4
separator = "." * INPUT_N

stream = separator.join(data)

words = sorted(list(set(stream)))

ctoi = {c: i for i, c in enumerate(words)}
itoc = {ctoi[c]: c for c in ctoi}

stream = [ctoi[c] for c in stream]

X = torch.tensor([stream[i : i + INPUT_N] for i in range(len(stream) - INPUT_N)])
Y = torch.tensor([stream[i + INPUT_N] for i in range(len(stream) - INPUT_N)])


# shuffle set
indexes = torch.randperm(len(X))
X = X[indexes]
Y = Y[indexes]

# split to train and val
ratio = 0.95
split_id = int(len(X) * ratio)
X_train = X[:split_id]
Y_train = Y[:split_id]

X_val = X[split_id:]
Y_val = Y[split_id:]

EMBEDDING_DIM = 16
HIDDEN_N = 256

model = nn.Sequential(
    nn.Embedding(len(words), EMBEDDING_DIM),
    nn.Flatten(),
    nn.Linear(INPUT_N * EMBEDDING_DIM, HIDDEN_N),
    nn.ReLU(),
    nn.Linear(HIDDEN_N, HIDDEN_N),
    nn.ReLU(),
    nn.Linear(HIDDEN_N, HIDDEN_N),
    nn.ReLU(),
    nn.Linear(HIDDEN_N, len(words))
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

print(sum(p.numel() for p in model.parameters()))

train_loss = []
val_loss = []

BATCH_SIZE = 4096
best_loss = 10
counter = 10
for i in range(256):
    indexes = torch.randint(0, len(X_train), (BATCH_SIZE, ))
    batch_X = X_train[indexes]
    batch_Y = Y_train[indexes]
    pred = model.forward(batch_X)
    optimizer.zero_grad()
    loss1 = criterion(pred, batch_Y)
    loss1.backward()
    optimizer.step()
    train_loss.append(loss1.item())

    # validation
    pred = model.forward(X_val)
    loss = criterion(pred, Y_val)
    print(f"Train loss: {loss1.item()}")
    print(f"Val loss: {loss.item()}\n")
    if best_loss > loss.item():
        best_loss = loss.item()
        counter = 4
    
    else:
        counter -= 1
        print(counter)
        if counter == 0:
            break
    

    val_loss.append(loss.item())

plt.plot(train_loss)
plt.plot(val_loss)
plt.show()

while 2137:
    text = "." * INPUT_N
    while 42:
        data_in = torch.tensor([ctoi[c] for c in text])
        data_in = torch.reshape(data_in, (1, -1))
        pred = model.forward(data_in)
        pred = torch.softmax(pred, dim=1)
        c = torch.multinomial(pred, 1, replacement=True)
        c = itoc[c.item()]
        if c == ".":
            break

        text = text[1:] + c
        print(c, end="")

    input()