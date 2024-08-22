Resnet:

![image](https://github.com/ymengvanessa/AI-ML-CV/blob/main/res.jpg)

![image](https://github.com/ymengvanessa/AI-ML-CV/blob/main/dual-att%20net.png)

- 优化器 `import torch.nn.functional as F`
- 网络结构由一个或者多个class实现，用`super(子类,self). __init__`继承nn.Module，初始化：
  ```
  network=nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
```
- encoder初始化方法：
```
self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
    
```
- decoder初始化方法：
```
embedding+gru+
self.out = nn.Linear(hidden_size, output_size)
```

前向传播方法中输入encoder的输出，first hidden state =last hidden state of encoder, embedding+relu+gru
```
    def forward(self, x):#input张量
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
```
torch.cat decoder的输出向量+softmax
模型训练：损失函数+优化器
```
loss=nn.具体损失函数名称()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```
- 计算图

- attention
初始化方法为三个nn.Linear生成的全连接层
attention类用query,key,F.softmax计算注意力权重，attention decoder类初始化为embedding,attention,gru，全连接层，dropout，
```
    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
```
