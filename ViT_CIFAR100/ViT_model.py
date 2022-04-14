import torch
from torch import nn


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_c, embed_size):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_size = embed_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_size, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_size))

    def forward(self, x):
        x = self.proj(x).reshape(x.size(0), self.embed_size, -1).transpose(1, 2)  #size:(b, num_patches, embed_size)
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], 1)
        x += self.pos_embed
        return x


class MLP(nn.Module):
    def __init__(self, embed_size, hidden_dim, drop_p=0) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_dim, embed_size),
            nn.Dropout(drop_p))

    def forward(self, x):
        return self.fc(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, attn_drop_p=0, fc_drop_p=0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.linear_Q = nn.Linear(embed_size, embed_size)
        self.linear_K = nn.Linear(embed_size, embed_size)
        self.linear_V = nn.Linear(embed_size, embed_size)
        self.multi_attn = nn.MultiheadAttention(embed_size, num_heads, attn_drop_p, batch_first=True)
        self.linear = nn.Linear(embed_size, embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, hidden_dim, fc_drop_p)


    def forward(self, x):
        x = self.norm1(x)
        Q, K, V = self.linear_K(x), self.linear_Q(x), self.linear_V(x)
        output, attn_weights = self.multi_attn(Q, K, V)
        output = self.mlp(self.norm2(output))
        return output


class ViT(nn.Module):
    def __init__(self,
                num_layers=4,
                img_size=224,
                patch_size=16,
                embed_size=768,
                num_heads=8,
                hidden_dim=2048, 
                num_classes=100,
                attn_drop_p=0,
                fc_drop_p=0,
                in_c=3):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_c, embed_size)
        self.encoder_layer = nn.Sequential(
            *nn.ModuleList([EncoderBlock(embed_size, num_heads, hidden_dim, attn_drop_p, fc_drop_p)]*num_layers)
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder_layer(x)
        output = self.mlp_head(x[:, 0, :])
        return output



if __name__ == '__main__':
    net = ViT(8, 72, 6, 64, 8, 128, 100)
    x = torch.rand(5, 3, 72, 72)
    y = net(x)
    print(y.size())