# Figure 2 Routing Compute Analysis

This directory contains the theoretical accounting used to plot the routing
compute overhead of sparse decode attention. Storage is intentionally not
plotted in this version.

## Setting

Unless stated otherwise, the figure uses a single active attention layer of
Llama-3.1-8B during single-token decode:

| Symbol | Meaning | Value |
|---|---:|---:|
| `S` | full KV cache length | `8192` |
| `H` | attention heads | `32` |
| `G` | KV heads | `8` |
| `D` | head dimension | `128` |
| `c` | compression ratio, `S / B` | `4, 8, 16, 32, 64` |
| `B` | selected-token budget, `S / c` | `2048, 1024, 512, 256, 128` |
| `P` | Quest page size | `16` |
| `r` | SparQ/Loki rank or channel budget | `16` |

The selected sparse attention compute is:

```math
C_{\mathrm{sparse}}(B) = 2HBD = 2H\frac{S}{c}D.
```

The factor `2` accounts for `qK^T` and `PV`. We omit lower-order softmax,
top-k/sort, masking, and cache-update costs. This makes the plot a clean
routing-vs-attention comparison rather than a kernel benchmark.

The plotted value is:

```math
\mathrm{RoutingComputeShare}
= \frac{C_{\mathrm{route}}}{C_{\mathrm{route}} + C_{\mathrm{sparse}}(B)}.
```

Equivalently, if `R = C_route / C_sparse`, then:

```math
\mathrm{RoutingComputeShare}
= \frac{R}{1 + R}.
```

## Method Formulas

### Quest

Quest stores page-wise elementwise `K_min` and `K_max` summaries, then scores
all pages with the current query and selects top pages.

```math
C_{\mathrm{route}}^{\mathrm{Quest}}
= H \frac{S}{P} D.
```

Relative to sparse selected attention:

```math
R_{\mathrm{Quest}}
= \frac{C_{\mathrm{route}}^{\mathrm{Quest}}}{C_{\mathrm{sparse}}}
= \frac{c}{2P}.
```

With `P=16`, Quest routing becomes increasingly dominant as compression
increases.

### SparQ

SparQ picks the top-`r` absolute query channels, scans those channels over the
full key cache to approximate attention scores, then performs exact attention
on the selected `B` tokens.

```math
C_{\mathrm{route}}^{\mathrm{SparQ}}
= HSr.
```

Relative to sparse selected attention:

```math
R_{\mathrm{SparQ}}
= \frac{rS}{2BD}
= \frac{cr}{2D}.
```

With `r=16, D=128`, this simplifies to `R = c / 16`.

### Loki

Loki ranks tokens using low-dimensional PCA-projected keys, then computes
full-dimensional exact attention on the selected `B` tokens. The plotted
formula follows the paper-level idealized analysis with cached projected keys:

```math
C_{\mathrm{route}}^{\mathrm{Loki}}
= H(Sr + 2D^2).
```

Relative to sparse selected attention:

```math
R_{\mathrm{Loki}}
= \frac{Sr + 2D^2}{2BD}
= \frac{cr}{2D} + \frac{cD}{S}.
```

The second term is the per-step PCA projection cost for the current query and
key under the paper-style `2D^2` model. In the script this can be changed with
`--loki-projection rank` or `--loki-projection none`.

### HSA

HSA uses full KV plus page-wise `K_min/K_max` summaries. Its default parameter
policy is:

```math
c = S/B,
```

```math
\rho = \min(0.2 + 0.06\log_2 c, 0.8),
```

```math
M = S / c^\rho,
```

```math
c_2 = M/B,
```

```math
p = \max(1, \min(\lfloor c_2 \rfloor, \lceil \sqrt{c_2} \rceil)),
```

```math
r_{\mathrm{hsa}}
= \min(D, \max(1, \mathrm{round}(D p / c_2))).
```

The router scores `S/p` page summaries using `r_hsa` channels:

```math
C_{\mathrm{route}}^{\mathrm{HSA}}
= H \frac{S}{p} r_{\mathrm{hsa}}.
```

Relative to sparse selected attention:

```math
R_{\mathrm{HSA}}
= \frac{c r_{\mathrm{hsa}}}{2pD}.
```

## Reproducing The Figure

From the repository root:

```bash
python analysis/figure2_router_overhead/plot_router_overhead.py
```

Outputs are written to:

```text
analysis/figure2_router_overhead/out/
```

The main combined figure:

- `routing_compute_vs_compression_llama31_8b_8k.png`
- `routing_compute_vs_compression_llama31_8b_8k.pdf`
- `routing_compute_vs_compression_llama31_8b_8k.csv`

Optional arguments:

```bash
python analysis/figure2_router_overhead/plot_router_overhead.py \
  --seq-len 8192 \
  --compression-ratios 4 8 16 32 64 \
  --out-dir analysis/figure2_router_overhead/out
```
