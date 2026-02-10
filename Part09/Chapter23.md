# 第23章 CPU フルスクラッチ実装（C++）

Part IX のフルスクラッチ実装のうち、**CPU 側** を C++ で実装します。第 22 章で定めた I/F（rasterizer_cpu_forward / rasterizer_cpu_backward）とデータ構造（types.h）に従い、**MVP 変換**（23.1）、**エッジ関数・重心座標によるラスタライザ本体**（23.2）、**深度バッファと可視性判定・属性補間**（23.3）、**逆伝播の完全手実装**（23.4）を順に組み立てます。数式は第 6 章・第 11 章、考慮事項は第 12 章を参照します。

---

## 23.1 MVP 変換の実装（Eigen 利用）

第 22 章で **math_util** は Eigen を用いると定めました。CPU ラスタライザでは、頂点をモデル空間からクリップ空間へ変換し、NDC を経てスクリーン座標に写すまでを **math_util** の API で行います。

### 23.1.1 処理の流れ

1. **モデル空間の頂点** $\mathbf{p}_{\text{model}} = (x, y, z, 1)^\top$ に、**モデル・ビュー・射影行列** $\mathbf{M}_{\text{MVP}}$（列優先 4×4）を掛けて **クリップ空間** の同次座標を得る。
   $$
   \mathbf{p}_{\text{clip}} = \mathbf{M}_{\text{MVP}} \, \mathbf{p}_{\text{model}} = (x_c, y_c, z_c, w_c)^\top
   $$
2. **NDC**: $x_{\text{NDC}} = x_c / w_c$, $y_{\text{NDC}} = y_c / w_c$, $z_{\text{NDC}} = z_c / w_c$。第 22 章の座標系の約束では NDC の範囲は $[-1, 1]$。
3. **スクリーン座標**: 第 22 章 22.3.6 に合わせ、$y$ 上向き・ピクセル中心を整数で表す写像にする。`math_util.h` の `ndc_to_screen` で、NDC $(x_{\text{NDC}}, y_{\text{NDC}})$ と解像度 $(W, H)$ からスクリーン座標 $(x_s, y_s)$ を計算する。

### 23.1.2 math_util の呼び出し

ラスタライザ側では、**全頂点** についてあらかじめクリップ座標とスクリーン座標を計算し、作業用バッファに保持します。

```cpp
// 疑似コード: 頂点をクリップ空間・スクリーン空間に変換
void transform_vertices(
    const Vertex* vertices,
    int num_vertices,
    const float* model_view_proj_4x4,
    int width, int height,
    float* out_clip_x, float* out_clip_y, float* out_clip_z, float* out_clip_w,
    float* out_screen_x, float* out_screen_y)
{
    for (int i = 0; i < num_vertices; ++i) {
        float pos[4] = { vertices[i].position[0], vertices[i].position[1],
                         vertices[i].position[2], 1.f };
        float clip[4];
        apply_mvp(clip, pos, model_view_proj_4x4);
        out_clip_x[i] = clip[0]; out_clip_y[i] = clip[1];
        out_clip_z[i] = clip[2]; out_clip_w[i] = clip[3];
        float ndc[2] = { clip[0]/clip[3], clip[1]/clip[3] };
        float screen[2];
        ndc_to_screen(screen, ndc, width, height);
        out_screen_x[i] = screen[0]; out_screen_y[i] = screen[1];
    }
}
```

- クリップ空間の $w_c$ は、のちの **透視補正**（$1/w$ 補間）で各頂点の $w_k$ として使います。$z_c/w_c$ は深度比較に使います。
- **可視性**: $w_c \le 0$ の頂点はカメラの後ろにあるため、その頂点を含む三角形はカリングするか、クリッピングで扱います。最小実装では「三角形の 3 頂点のいずれかが $w_c \le 0$ ならスキップ」とする簡易策でもよいです（第 12 章 12.1.3）。

---

## 23.2 エッジ関数・重心座標によるラスタライザ本体

第 6 章 6.2–6.3 の式をそのまま C++ で実装します。

### 23.2.1 エッジ関数と面積

三角形の 3 頂点を **スクリーン座標** の 2D 点で $\mathbf{p}_0 = (x_0, y_0)$, $\mathbf{p}_1 = (x_1, y_1)$, $\mathbf{p}_2 = (x_2, y_2)$ とし、**反時計回り** に並んでいるものとします。点 $\mathbf{q} = (x, y)$ について、

$$
E_{01}(\mathbf{q}) = (x - x_0)(y_1 - y_0) - (y - y_0)(x_1 - x_0)
$$
$$
E_{12}(\mathbf{q}) = (x - x_1)(y_2 - y_1) - (y - y_1)(x_2 - x_1)
$$
$$
E_{20}(\mathbf{q}) = (x - x_2)(y_0 - y_2) - (y - y_2)(x_0 - x_2)
$$

符号付き面積の 2 倍は $2A = E_{12}(\mathbf{p}_0)$ です。反時計回りなら $2A > 0$ です。

```cpp
inline float edge_function(float x, float y,
    float x0, float y0, float x1, float y1)
{
    return (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0);
}
```

### 23.2.2 内外判定と重心座標

点 $\mathbf{q}$ が三角形の **内側または辺上** にある必要十分条件は、

$$
E_{01}(\mathbf{q}) \ge 0 \quad \text{かつ} \quad E_{12}(\mathbf{q}) \ge 0 \quad \text{かつ} \quad E_{20}(\mathbf{q}) \ge 0
$$

重心座標は $u = b_1 = E_{20}(\mathbf{q}) / (2A)$, $v = b_2 = E_{01}(\mathbf{q}) / (2A)$, $b_0 = 1 - u - v$ です。

- **退化三角形**: $|2A| < \varepsilon$ のときは 0 除算を避けるため、その三角形は描画しない（第 12 章 12.1.1）。実装では `const float eps = 1e-7f` などとし、`fabsf(two_a) < eps` なら continue する。
- **背面カリング**: 第 12 章 12.1.2 の通り、$2A < 0$ なら背面とみなしスキップするか、オプションで両面描画にする。

### 23.2.3 バウンディングボックスとピクセル走査

三角形をラスタライズするとき、**スクリーン座標での軸並行バウンディングボックス（AABB）** を計算し、その範囲内のピクセルだけを走査します。

- ピクセル中心を $\mathbf{q} = (i + 0.5, j + 0.5)$ とします（$i, j$ は整数）。第 22 章の座標系では $y=0$ が上端なので、$j$ は 0 ～ height-1 で上から下へ対応させます。
- AABB は $\min(x_0,x_1,x_2)$, $\max(x_0,x_1,x_2)$, $\min(y_0,y_1,y_2)$, $\max(y_0,y_1,y_2)$ から得られます。解像度内にクリップし、その範囲で二重ループします。

```cpp
int x_min = (int)floorf(fminf(fminf(x0, x1), x2));
int x_max = (int)ceilf(fmaxf(fmaxf(x0, x1), x2));
int y_min = (int)floorf(fminf(fminf(y0, y1), y2));
int y_max = (int)ceilf(fmaxf(fmaxf(y0, y1), y2));
x_min = max(0, min(x_min, width - 1));
x_max = max(0, min(x_max, width - 1));
y_min = max(0, min(y_min, height - 1));
y_max = max(0, min(y_max, height - 1));
for (int j = y_min; j <= y_max; ++j)
    for (int i = x_min; i <= x_max; ++i) {
        float qx = (float)i + 0.5f, qy = (float)j + 0.5f;
        float e01 = edge_function(qx, qy, x0, y0, x1, y1);
        float e12 = edge_function(qx, qy, x1, y1, x2, y2);
        float e20 = edge_function(qx, qy, x2, y2, x0, y0);
        if (e01 >= 0 && e12 >= 0 && e20 >= 0) {
            float u = e20 / two_a, v = e01 / two_a;
            // 深度を計算し、深度テスト（23.3）へ
        }
    }
```

---

## 23.3 深度バッファと可視性判定、属性補間（パースペクティブ補正）

### 23.3.1 深度の計算と深度テスト

各ピクセルで、重心座標 $(u, v)$ が得られたら、**深度** を補間します。第 6 章 6.4 の通り、透視の下では $z/w$ と $1/w$ を線形補間してから $z = (z/w)/(1/w)$ で復元するのが正しいです。実装では、3 頂点のクリップ空間の $z_c$, $w_c$ を使って

$$
\frac{z}{w} = (1-u-v)\frac{z_0}{w_0} + u\frac{z_1}{w_1} + v\frac{z_2}{w_2}
$$

を計算し、これを「正規化深度」として幾何バッファに書き、**深度テスト** では「より小さい（手前）を勝たせる」ようにします（第 22 章の約束）。  
初期化時には深度バッファを「奥」の値（例: 1.0）でクリアし、各三角形を処理するとき、現在ピクセルの深度が既存より手前なら上書きし、幾何バッファ（三角形 ID、$u$, $v$、深度）と深度バッファを更新します。

- **正確性（精度）**: 深度バッファと「手前なら上書き」を徹底すれば、**三角形の処理順序に依存せず**、各ピクセルには最終的に一番手前の三角形だけが残る。したがって、順方向の画像も逆伝播で勾配が流れる「見えている三角形」も、描画順に依存せず正しい。
- **パフォーマンス**: 描画順は **性能に影響する**。**渡された順**のまま描画すると、奥の三角形を先にラスタライズして書き、のちに手前の三角形で上書きされることが多く、**オーバードロー**（いったん書いたフラグメントが捨てられる）が増える。重なりが多いメッシュでは、無駄なフラグメント計算が増え、CPU ラスタライザでも負荷が大きくなる。  
  **推奨**: 三角形が多数かつ重なりが大きい場合は、**深度でソート**（例: 各三角形の重心や最小深度で前から順）してから描画すると、手前の三角形を先に書くことで奥の三角形が深度テストで弾かれ、オーバードローを減らせる。最小実装では「渡された順」でも正しい結果は得られるが、パフォーマンスが必要なら前処理でソートするか、第 24 章の GPU 実装（アトミック深度ソート等）を参照する。

### 23.3.2 幾何バッファの書き込み

ピクセル $(i, j)$ が三角形 `tri_id` の内側にあり、深度テストに勝ったとき、**幾何バッファ** の該当要素に以下を書き込みます（第 22 章 22.3.6 の GeometryPixel）。

- `u`, `v`: 上で計算した重心座標。
- `depth`: 正規化深度 $z/w$（または比較用に別の表現でもよい。一貫させればよい）。
- `tri_id`: 三角形の番号（1 始まりにして 0 を「背景」にするか、0 始まりで「背景」を -1 や 0 で表すかは設計による）。ここでは 1 始まりとし、三角形 0 は `tri_id == 1` とする。

線形インデックスは `idx = j * width + i` です。

### 23.3.3 属性補間（色・UV・法線）とパースペクティブ補正

**ラスタライズ** が終わったら、**幾何バッファ**（各ピクセルの三角形 ID、$u$, $v$）を使って、**フレームバッファ** に色などを書き込みます。第 6 章 6.4.2 の **$1/w$ 補間** を使います。

頂点 $k$ の $w_k$ はクリップ空間の $w_c$、属性を $A_k$（色の R 成分、G、B、UV、法線など）とすると、

$$
\frac{1}{w} = \frac{1-u-v}{w_0} + \frac{u}{w_1} + \frac{v}{w_2}, \quad
\frac{A}{w} = \frac{(1-u-v)A_0}{w_0} + \frac{u A_1}{w_1} + \frac{v A_2}{w_2}
$$

$$
A = \frac{A/w}{1/w}
$$

で補間します。色の RGB 各成分、UV、法線の各成分にこの式を適用します。FramePixel には `color[3]` と `alpha` を入れるとします（第 22 章）。背景ピクセル（`tri_id == 0`）は既定値（例: 0）で埋めます。

```cpp
// 疑似コード: 1 ピクセル分の属性補間（透視補正）
float inv_w = (1.f - u - v) / w0 + u / w1 + v / w2;
float w_recip = 1.f / inv_w;
float color_r = ((1.f - u - v) * c0.r / w0 + u * c1.r / w1 + v * c2.r / w2) * w_recip;
// color_g, color_b 同様。frame_buffer[idx].color[0/1/2], alpha を設定。
```

---

## 23.4 逆伝播の完全手実装（解析的勾配）

自動微分ライブラリは使わず、第 6 章・第 11 章の **解析的勾配** を式の通りに C++ で実装します。

### 23.4.1 逆伝播の流れ（概要）

上流から **フレームバッファの各ピクセル・各チャネルに対する勾配** $\partial L / \partial A$（$A$ は補間された色など）が `dL_d_frame` で渡されます（第 22 章 22.3.6）。これを次の順で伝播させます。

1. **補間式の逆伝播**: $\partial L / \partial A$ から $\partial L / \partial (A/w)$, $\partial L / \partial (1/w)$ を求め、さらに $\partial L / \partial u$, $\partial L / \partial v$ および **頂点属性** $A_k$ と **頂点の $w_k$** への勾配を求める（第 6 章 6.6.3）。
2. **重心座標からスクリーン座標へ**: $\partial L / \partial u$, $\partial L / \partial v$ から、頂点の **スクリーン座標** $(x_k, y_k)$ への勾配 $\partial L / \partial x_k$, $\partial L / \partial y_k$ を、エッジ関数・面積の偏微分で計算する（第 6 章 6.3.2）。
3. **スクリーン座標からクリップ空間へ**: NDC は $x_c/w_c$ などの除算で得られるため、除算の逆伝播（第 6 章 6.6.2）で、クリップ空間の $(x_c, y_c, z_c, w_c)_k$ への勾配を求める。
4. **クリップ空間からモデル空間へ**: クリップ座標は MVP の出力なので、**MVP の転置**（または backward 用の行列）を掛けて、モデル空間の頂点位置への勾配に変換する。
5. **頂点ごとに蓄積**: 複数のピクセルから同じ頂点に勾配が流れるため、`vertex_grad` に **加算** する。呼び出し側でゼロ初期化済みであることを前提とする。

### 23.4.2 補間式の逆伝播（1 ピクセル）

第 6 章 6.6.3 の式を使います。$\lambda_0 = 1-u-v$, $\lambda_1 = u$, $\lambda_2 = v$ とし、$A = (A/w)/(1/w)$ なので、

$$
\frac{\partial L}{\partial (A/w)} = \frac{\partial L}{\partial A} \cdot w, \qquad
\frac{\partial L}{\partial (1/w)} = -\frac{\partial L}{\partial A} \cdot A \cdot w
$$

さらに $\partial (A/w) / \partial \lambda_k = A_k/w_k$, $\partial (1/w) / \partial \lambda_k = 1/w_k$ などから、

$$
\frac{\partial L}{\partial u} = \frac{\partial L}{\partial (A/w)} \left( \frac{A_1}{w_1} - \frac{A_0}{w_0} \right) + \frac{\partial L}{\partial (1/w)} \left( \frac{1}{w_1} - \frac{1}{w_0} \right)
$$

$$
\frac{\partial L}{\partial v} = \frac{\partial L}{\partial (A/w)} \left( \frac{A_2}{w_2} - \frac{A_0}{w_0} \right) + \frac{\partial L}{\partial (1/w)} \left( \frac{1}{w_2} - \frac{1}{w_0} \right)
$$

色が RGB 3 チャネルなら、各チャネルで上記を計算し、$\partial L / \partial u$, $\partial L / \partial v$ を **合計** する（同じ $u$, $v$ が色の 3 成分に効くため）。  
頂点属性への勾配は、$\partial L / \partial (A/w)$ と $\partial L / \partial (1/w)$ から $\partial L / \partial A_k$ を求め、`vertex_grad[k].d_color` などに加算する（第 6 章 6.6.3 の $\lambda_k/w_k$, $- \lambda_k A_k / w_k^2$ 等）。

### 23.4.3 重心座標からスクリーン座標への勾配

第 6 章 6.3.2 の通り、$u = E_{20}/(2A)$, $v = E_{01}/(2A)$ を各頂点のスクリーン座標で偏微分する。例えば

$$
\frac{\partial E_{20}}{\partial x_0} = y - y_2, \quad \frac{\partial E_{20}}{\partial y_0} = -(x - x_2), \quad \ldots
$$

$2A = E_{12}(\mathbf{p}_0)$ なので $\partial (2A) / \partial x_0$ なども計算でき、商の微分則で $\partial u / \partial x_k$, $\partial u / \partial y_k$, $\partial v / \partial x_k$, $\partial v / \partial y_k$ が得られる。これらと $\partial L / \partial u$, $\partial L / \partial v$ から、

$$
\frac{\partial L}{\partial x_k} = \frac{\partial L}{\partial u} \frac{\partial u}{\partial x_k} + \frac{\partial L}{\partial v} \frac{\partial v}{\partial x_k}, \quad
\frac{\partial L}{\partial y_k} = \frac{\partial L}{\partial u} \frac{\partial u}{\partial y_k} + \frac{\partial L}{\partial v} \frac{\partial v}{\partial y_k}
$$

を計算し、スクリーン座標の頂点 $k$ への勾配とする。

### 23.4.4 スクリーン座標からクリップ空間へ

スクリーン座標は NDC の線形写像なので、$x_s = a \cdot x_{\text{NDC}} + b$ の形。したがって $\partial L / \partial x_{\text{NDC}} = \partial L / \partial x_s \cdot \partial x_s / \partial x_{\text{NDC}}$ は容易。NDC は $x_{\text{NDC}} = x_c / w_c$ なので、第 6 章 6.6.2 の除算の逆伝播により、

$$
\frac{\partial L}{\partial x_c} \mathrel{+}= \frac{\partial L}{\partial x_{\text{NDC}}} \frac{1}{w_c}, \quad
\frac{\partial L}{\partial w_c} \mathrel{+}= \frac{\partial L}{\partial x_{\text{NDC}}} \left( -\frac{x_c}{w_c^2} \right)
$$

（$y$, $z$ も同様。）各頂点 $k$ について、$(x_c, y_c, z_c, w_c)_k$ への勾配が揃う。

### 23.4.5 クリップ空間からモデル空間（MVP の逆伝播）

クリップ座標 $\mathbf{p}_{\text{clip}} = \mathbf{M}_{\text{MVP}} \mathbf{p}_{\text{model}}$ なので、逆伝播では

$$
\frac{\partial L}{\partial \mathbf{p}_{\text{model}}} = \mathbf{M}_{\text{MVP}}^\top \frac{\partial L}{\partial \mathbf{p}_{\text{clip}}}
$$

を用いる。同次座標の第 4 成分は 1 固定なので、実装では 4 成分目への勾配は使わない（または無視）でよい。得られた 3 成分を `vertex_grad[k].d_position` に加算する。

### 23.4.6 rasterizer_cpu_backward の疑似コード構造

```text
1. 全ピクセルについて geometry を読む（tri_id, u, v, depth）。
2. tri_id == 0（背景）のピクセルはスキップ。
3. そのピクセルに対応する三角形の 3 頂点インデックスと、その頂点の
   クリップ座標・スクリーン座標・属性・w を取得。
4. dL_d_frame からそのピクセルの ∂L/∂(R,G,B,alpha) を取得。
5. 補間の逆伝播で ∂L/∂u, ∂L/∂v と ∂L/∂(各頂点の属性) を計算。
   → vertex_grad の d_color 等に加算。
6. u,v のスクリーン座標に関する微分で ∂L/∂(x_k,y_k) を計算。
7. スクリーン→NDC→クリップの逆伝播で ∂L/∂(x_c,y_c,z_c,w_c)_k を計算。
8. MVP の転置で ∂L/∂(モデル空間頂点) を計算。
   → vertex_grad の d_position に加算。
```

- **シルエット勾配**（第 11 章）: 境界ピクセルでは、エッジの移動に応じた勾配を追加すると、形状最適化がより安定する。最小実装では「内部の補間勾配のみ」でも動作する。必要なら境界検出（隣接ピクセルと tri_id が異なる等）を入れ、シルエット勾配を加算する。

---

## 23.5 まとめと第 24 章への接続

- **23.1**: MVP 変換は math_util（Eigen）の `apply_mvp` と `ndc_to_screen` で行い、クリップ座標・スクリーン座標を全頂点分用意する。
- **23.2**: エッジ関数 $E_{01}$, $E_{12}$, $E_{20}$ と $2A$ で内外判定と重心座標 $u$, $v$ を計算。退化三角形（$|2A| < \varepsilon$）はスキップ、背面はオプションでカリング。
- **23.3**: 深度は $z/w$ の線形補間で計算し、深度テストで手前を残す。幾何バッファに tri_id, u, v, depth を書き、そのあと $1/w$ 補間で色・UV・法線をフレームバッファに書き込む。
- **23.4**: 逆伝播は補間→$u,v$→スクリーン座標→クリップ空間→モデル空間の順で、第 6 章の式を手で実装し、`vertex_grad` に加算する。

この CPU 実装は、第 22 章の I/F とデータ構造に従っており、第 24 章の GPU（D3D12 コンピュートシェーダー）実装と **同じ入力** で **同じ出力形式** を生成する。第 25 章で CPU と GPU の結果を比較・検証する際の基準になる。
