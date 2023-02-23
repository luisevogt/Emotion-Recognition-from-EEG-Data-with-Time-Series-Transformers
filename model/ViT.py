import numpy as np
import torch
import torch.nn as nn


def patchify(images, n_patches):
    # TODO: don't use loops here
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size,
                        j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()

    return patches


def get_positional_encoding(seq_lenght, d):
    result = torch.ones(seq_lenght, d)
    for i in range(seq_lenght):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))

    return result


class ViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(ViT, self).__init__()

        # Attributes
        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Inputs and patch sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches."
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches."
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable Classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_encoding(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False  # not learned

        # 4) Encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(self.hidden_d, self.n_heads) for _ in range(self.n_blocks)])

        # 5) Classification
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed

        for block in self.blocks:
            out = block(out)

        # Classification token
        out = out[:, 0]

        return self.mlp(out)


class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()

        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads."

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequences):
        # Sequences have shape (N, seq_length, token dim)
        # We go into shape (N, seq length, n heads, token dim / n heads)
        # And come back to (N, seq_length, item _dim)(through concatenation)
        # TODO: don't use loops here
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class ViTBlock(nn.Module):
    # Encoder block
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm = nn.LayerNorm(hidden_d)
        self.msa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.msa(self.norm(x))
        out = out + self.mlp(self.norm2(out))

        return out


if __name__ == "__main__":
    # Encoder
    model = ViTBlock(hidden_d=8, n_heads=2)
    x = torch.randn(7, 50, 8)  # Fake seqs
    print(model(x).shape)
