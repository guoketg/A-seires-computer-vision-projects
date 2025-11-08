import torch
from torch import nn,optim
from utils import train,load_data_cifar10

class PatchEmbedding(nn.Module):
    def __init__(self,image_size=224,patch_size=16,in_channels=3,embed_dim=768):
        super().__init__()
        self.image_size=image_size
        self.patch_size=patch_size
        self.n_patches=(image_size//patch_size)**2

        self.proj=nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self,x):
        x=self.proj(x)
        # (batch_size, channels, height, width) -> (batch_size, embed_dim, n_patches^(1/2), n_patches^(1/2))
        x=x.flatten(2)
        # (batch_size, embed_dim, n_patches)
        x=x.transpose(1,2)
        # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embed_dim=768,n_heads=12,dropout=0.0):
        super().__init__()
        self.embed_dim= embed_dim
        self.n_heads=n_heads
        self.head_dim=embed_dim//n_heads

        assert (self.head_dim*n_heads==embed_dim),"Embedding dimension must be divisible by number of heads"

        self.qkv=nn.Linear(embed_dim,embed_dim*3)
        self.attn_drop=nn.Dropout(dropout)
        self.proj=nn.Linear(embed_dim,embed_dim)
        self.proj_drop=nn.Dropout(dropout)
        #缩放因子
        self.scale=self.head_dim**-0.5

    def forward(self,x):
        # 生成qkv (batch_size, n_patches, embed_dim*3)
        #head_dim*n_heads=embed_dim
        batch_size,n_patches,embed_dim=x.shape

        qkv=self.qkv(x)
        # 重塑为 (batch_size, n_patches, 3, n_heads, head_dim)
        qkv=qkv.reshape(batch_size,n_patches,3,self.n_heads,self.head_dim)
        # 置换维度为 (3, batch_size, n_heads, n_patches, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 修复：正确的维度置换

        q,k,v=qkv[0],qkv[1],qkv[2]

        #计算注意力分数
        attn=(q@k.transpose(-2,-1))*self.scale
        attn=attn.softmax(dim=-1)
        attn=self.attn_drop(attn)

        #注意力运用到值上
        out=attn@v
        # 转置和重塑 (batch_size, n_patches, n_heads, head_dim) -> (batch_size, n_patches, embed_dim)
        out=out.transpose(1,2).reshape(batch_size,n_patches,embed_dim)

        #输出投影
        out=self.proj(out)
        out=self.proj_drop(out)

        return out

class MLP(nn.Module):
    def __init__(self,in_features,hiddens_features=None,out_features=None,dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hiddens_features=hiddens_features or in_features

        self.fc1=nn.Linear(in_features,hiddens_features)
        self.act=nn.GELU()
        self.fc2=nn.Linear(hiddens_features,out_features)
        self.drop=nn.Dropout(dropout)

    def forward(self,x):
        out1=self.act(self.fc1(x))
        return self.drop(self.fc2(out1))

class Block(nn.Module):
    def __init__(self,embed_dim=768,n_heads=12,mlp_ratio=4.0,dropout=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(embed_dim)
        self.attn=MultiHeadSelfAttention(embed_dim,n_heads,dropout)
        self.norm2=nn.LayerNorm(embed_dim)

        mlp_hidden_dim=int(embed_dim*mlp_ratio)
        self.mlp=MLP(
            in_features=embed_dim,
            hiddens_features=mlp_hidden_dim,
            dropout=dropout
        )

    #残差连接
    def forward(self,x):
        x=x+self.attn(self.norm1(x))
        x=x+self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):

    def __init__(self,image_size=224,patch_size=16,in_channels=3,n_classes=1000,
                 embed_dim=768,depth=12,n_heads=12,mlp_ratio=4.0,dropout=0.0,emb_dropout=0.0):
        super().__init__()
        self.patch_embed=PatchEmbedding(image_size,patch_size,in_channels,embed_dim)
        n_patches=self.patch_embed.n_patches
        """
        第一个维度 1：表示批量维度（batch size）的占位符。
        在 PyTorch 中，参数通常会预留批量维度，方便后续通过广播机制与输入数据（带有批量维度）进行运算。
        第二个维度 1：表示分类标记的数量。ViT 中通常只需要 1 个分类标记（类似 BERT 中的 [CLS]），
        它会被拼接到 patch 序列的开头。
        第三个维度 embed_dim：表示每个标记的嵌入维度（与 patch 的嵌入维度一致）。
        这是模型的核心维度，所有特征都会映射到这个维度空间进行处理。
        """
        #可学习的位置编码
        self.pos_embed=nn.Parameter(torch.zeros(1,n_patches+1,embed_dim))
        #可学习的分类token
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.dropout=nn.Dropout(emb_dropout)

        self.blocks=nn.ModuleList([
            Block(embed_dim,n_heads,mlp_ratio,dropout) for _ in range(depth)
        ])

        #层归一化
        self.norm=nn.LayerNorm(embed_dim)
        #分类头
        self.head=nn.Linear(embed_dim,n_classes)

        #初始化权重
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        self.apply(self._init_weights)


    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.trunc_normal_(m.weight,std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)

    def forward(self,x):
        batch_size=x.shape[0]
        x=self.patch_embed(x)

        #添加分类token
        cls_tokens=self.cls_token.expand(batch_size,-1,-1)
        x=torch.cat((cls_tokens,x),dim=1)

        #添加位置编码
        x=x+self.pos_embed
        x=self.dropout(x)

        for block in self.blocks:
            x=block(x)

        x=self.norm(x)
        cls_output=x[:,0]
        out=self.head(cls_output)

        return out

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=VisionTransformer(image_size=96,patch_size=4,in_channels=3,n_classes=10,
                          embed_dim=256,depth=4,n_heads=4,mlp_ratio=4.0,dropout=0.1,emb_dropout=0.1)
    net=net.to(device)
    total_params=sum(p.numel() for p in net.parameters())
    print(f"number of parameters:{total_params}")
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 10
    batch_size = 128
    lr = 1e-4
    path = "D:\code\python\data\CIFAR10"

    # 权重衰退 L2惩罚项
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_loader,test_loader=load_data_cifar10(batch_size,path,resize=96)
    train(net,train_loader,test_loader,num_epochs,loss_fn,optimizer,device)

    torch.save(net.state_dict(),'D:\code\python\save_position\\trained_VisionTransformer.pth')