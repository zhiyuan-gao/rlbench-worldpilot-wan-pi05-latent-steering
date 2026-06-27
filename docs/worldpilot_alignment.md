# WorldPilot Alignment Notes

本 repo 对齐的是 WorldPilot 的 Latent Steering 机制，不是 WorldPilot 的整体工程。

## WorldPilot Public Code Pattern

WorldPilot 的 image latent steering 逻辑可以概括成：

```text
future_image_latents: (B, N_cam, C, H_lat, W_lat)
flat per camera:      (B, N_cam, C * H_lat * W_lat)
projected tokens:     (B, N_cam, H_vlm)
vlm hidden states:    (B, L, H_vlm)
cross-attn output:    (B, L, H_vlm)
```

关键点：

- latent 是 VAE/video latent，不是 world model transformer block hidden state。
- fusion 不增加 VLM token 长度；它通过 cross-attn residual 改写原 VLM hidden states。
- projector 的输入维度由 latent spatial size 决定，输出维度对齐 VLM hidden dim。

## WAN Version

我们的 WAN future video model 预计提供 VAE-before-decode future video latent：

```text
(B, V, C, T_lat, H_lat, W_lat)
```

其中：

- `V`: RLBench views，默认 `front/left_shoulder/right_shoulder`
- `C`: WAN VAE latent channels
- `T_lat`: WAN latent-time positions
- `H_lat, W_lat`: WAN VAE latent spatial size

最接近 Cosmos-Predict future-video latent steering 的初始方式是：

```text
preserve T_lat:
  (B, V, C, T_lat, H_lat, W_lat)
  -> (B, V * T_lat, C * H_lat * W_lat)
  -> (B, V * T_lat, H_pi05)
```

这避免一开始就只取最后 latent step，也避免把整段未来视频平均成一个 token。`last` 和 `mean` 应作为 ablation，而不是默认假设。

## Non-Goals

- 不使用 Wan block13 hidden tokens。
- 不把 RGB decoded future video 输入 action policy。
- 不重新定义 pi0.5 action target。
- 不让 WorldPilot repo 成为主训练工程。

