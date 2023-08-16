import torch
import torch.nn as nn

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

class WordGenerator(nn.Module):
    def __init__(self, *args, dictionary_len, input_n, embedding_dim, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Embedding = nn.Embedding(dictionary_len, embedding_dim)

        self.Lin1 = nn.Linear(embedding_dim * input_n, 256)
        self.Lin2 = nn.Linear(256, 256)
        self.Lin3 = nn.Linear(256, 256)
        self.Lin4 = nn.Linear(256, dictionary_len)

    def forward(self, X):
        X = self.Embedding(X)

        X = nn.Flatten()(X)

        X = self.Lin1(X)
        X = nn.ReLU()(X)
        X = self.Lin2(X)
        X = nn.ReLU()(X)
        X = self.Lin3(X)
        X = nn.ReLU()(X)
        X = self.Lin4(X)

        return X

model = WordGenerator(dictionary_len=len(words), input_n=INPUT_N, embedding_dim=8)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

print(sum(p.numel() for p in model.parameters()))
for i in range(1024):
    indexes = torch.randint(0, len(X), (1024, ))
    batch_X = X[indexes]
    batch_Y = Y[indexes]
    pred = model.forward(batch_X)
    optimizer.zero_grad()
    loss = criterion(pred, batch_Y)
    loss.backward()
    optimizer.step()
    print(loss.item())

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