"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # x: [B, N, D] -> 拆分为 [B, N, H, D/H]，再交换 token 轴与 head 轴得到 [B, H, N, D/H]
        batch_size, token_count, _ = x.shape
        reshaped = x.reshape(batch_size, token_count, self.heads, self.head_dim)
        return reshaped.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q, k: [B, H, N, head_dim]；仅交换 k 的最后两轴为 [B, H, head_dim, N]，得到 scores: [B, H, N, N]
        scale = np.sqrt(np.float32(self.head_dim))
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # coefficients @ v: [B, H, N, head_dim] -> 转置回 [B, N, H, head_dim] -> 合并为 [B, N, D]
        context = coefficients @ v
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.width)

        # TODO-03c: output projection, using out_weight and out_bias.
        # out_weight 按 [out, in] = [D, D] 存储，因此乘以 out_weight.T 后加偏置，输出 [B, N, D]
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        # 第一条残差分支：将注意力更新量加回归一化前的输入 x，保持 [B, N, D]
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B, N, 4D] (图像塔为 3072)，mlp_update: [B, N, D] (图像塔为 768)
        hidden = u @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
        # 第二条残差分支：加回第一阶段残差结果 residual，输出 [B, N, D]
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # 输入 images: [B, C=3, H=224, W=224]，patch 边长为 32，网格为 7x7
        batch_size, channels, height, width = images.shape
        patch_size = 32
        grid_h = height // patch_size
        grid_w = width // patch_size
        # 1) 拆分网格轴与块内像素轴：[B, C, 7, 32, 7, 32]
        patches = images.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
        # 2) 重排为 [B, grid_h, grid_w, C, 32, 32]，保证网格先行后列（从左到右、从上到下），块内按 C,H,W 展开
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        # 3) 展平为 [B, 49, 3072]
        return patches.reshape(batch_size, grid_h * grid_w, channels * patch_size * patch_size)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        # patch_weight 形状为 [768, 3072]（[out, in]），无偏置项，右乘转置得到 [B, 49, 768]
        patch_tokens = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # 沿 token 轴 (axis=1) 在最前面拼接 CLS 得到 [B, 50, 768]，再广播加上位置编码 [50, 768]
        tokens = np.concatenate([cls, patch_tokens], axis=1)
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # 取出第 0 个位置上的 CLS token：[B, 50, 768] -> [B, 768]
        cls_tokens = tokens[:, 0, :]
        # 经输出层 LayerNorm 后与 self.projection ([768, 512]，已是 [in, out] 布局，无需转置) 相乘 -> [B, 512]
        output = self.post_norm(cls_tokens) @ self.projection

        return unit_vector(output)
