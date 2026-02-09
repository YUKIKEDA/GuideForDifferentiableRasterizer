# 第6章 三角形とスクリーン空間

Part III では **ラスタライゼーションの数学** を厳密に扱います。本章はその基盤で、三角形をスクリーン空間に投影したあと、**どのピクセルに属するか**（エッジ関数・重心座標）と **深度・属性をどう補間するか** を式で整理します。さらに **パースペクティブ補正** と、その **完全な微分**（逆伝播で必要な勾配）を導出します。6.6 の「スクリーン空間の重心座標から 3D 空間への変換勾配」は、微分可能ラスタライザで初心者が最もつまずくポイントなので、連鎖律を段階的に追います。

---

## 6.1 クリッピングと NDC

### 6.1.1 クリップ空間まで

第 2 章の通り、頂点は MVP により **クリップ空間** の同次座標まで変換されます。

$$
\mathbf{p}_{\text{clip}} = \mathbf{M}_{\text{proj}} \mathbf{M}_{\text{view}} \mathbf{M}_{\text{model}} \mathbf{p}_{\text{model}} = (x_c,\, y_c,\, z_c,\, w_c)^\top
$$

透視射影では $w_c > 0$ が可視領域（カメラの前方）に対応します。視錐台の外にある図形は **クリッピング** により、錐台と交差する部分だけを残すか、破棄します。クリッピングの詳細は実装編（Part V）に譲り、ここでは **クリップ空間で有効な三角形** が与えられたときの続きを考えます。

### 6.1.2 NDC とスクリーン座標

**正規化デバイス座標（NDC）** は、$w_c$ で割った座標です。

$$
x_{\text{NDC}} = \frac{x_c}{w_c}, \quad y_{\text{NDC}} = \frac{y_c}{w_c}, \quad z_{\text{NDC}} = \frac{z_c}{w_c}
$$

通常、可視領域は $x_{\text{NDC}}, y_{\text{NDC}}, z_{\text{NDC}} \in [-1, 1]$ の範囲です。**スクリーン（ウィンドウ）座標** は、解像度 $W \times H$ に対して例えば次のように取ります。

$$
x_s = \frac{W}{2}(1 + x_{\text{NDC}}), \quad y_s = \frac{H}{2}(1 - y_{\text{NDC}})
$$

$y$ を反転しているのは、NDC の $y$ 上向きとスクリーンの $y$ 下向きの慣習の違いによるものです。ピクセル $(i, j)$ の **中心** は多くの API で $(i + 0.5, j + 0.5)$ とします。  
ラスタライザの入力は、三角形の 3 頂点の **クリップ空間** の値 $(x_c, y_c, z_c, w_c)_k$（$k=0,1,2$）です。これから NDC・スクリーン座標を計算し、2D の $(x_s, y_s)$ 上でエッジ関数と重心座標を扱います。

### 6.1.3 $w$ の意味

透視射影では、$w_c$ はカメラ空間の深度（またはその線形関数）と一致するように作られます。したがって **奥の頂点ほど $w_c$ が大きい** です。のちの **$1/w$ 補間** は、この $w_c$（または NDC に写したあとの対応する $w$）を使って、正しい透視の下で属性を補間するためのものです。

---

## 6.2 エッジ関数と内外判定

### 6.2.1 2D エッジ関数の定義

三角形の 3 頂点を **スクリーン座標** の 2D 点で $\mathbf{p}_0 = (x_0, y_0)$, $\mathbf{p}_1 = (x_1, y_1)$, $\mathbf{p}_2 = (x_2, y_2)$ とし、**反時計回り** に並んでいるものとします。エッジ $\mathbf{p}_i \to \mathbf{p}_j$ に対して、点 $\mathbf{q} = (x, y)$ の **エッジの左側** にあるかどうかは、符号付き面積の 2 倍で判定できます。

$$
E_{ij}(\mathbf{q}) = (x - x_i)(y_j - y_i) - (y - y_i)(x_j - x_i)
$$

- $\mathbf{q}$ がエッジの **左側** なら $E_{ij}(\mathbf{q}) > 0$
- **右側** なら $E_{ij}(\mathbf{q}) < 0$
- **線上** なら $E_{ij}(\mathbf{q}) = 0$

三角形の **符号付き面積の 2 倍** は、例えば

$$
2A = E_{12}(\mathbf{p}_0) = (x_0 - x_1)(y_2 - y_1) - (y_0 - y_1)(x_2 - x_1)
$$

で与えられます。反時計回りなら $2A > 0$ です。

### 6.2.2 内外判定

点 $\mathbf{q}$ が三角形の **内部または辺上** にある必要十分条件は、3 本のエッジすべてで「左側」以上にあることです。

$$
E_{01}(\mathbf{q}) \ge 0 \quad \text{かつ} \quad E_{12}(\mathbf{q}) \ge 0 \quad \text{かつ} \quad E_{20}(\mathbf{q}) \ge 0
$$

ラスタライザでは、各ピクセル中心 $\mathbf{q} = (i+0.5, j+0.5)$ についてこの 3 つを計算し、すべて $\ge 0$ ならそのピクセルを三角形に属するとみなします（等号はエッジ上の扱いで、実装では片側に含める）。

### 6.2.3 エッジ関数と重心座標の対応

小三角形の符号付き面積の 2 倍を使うと、重心座標は次のように書けます。

$$
b_0 = \frac{E_{12}(\mathbf{q})}{2A}, \quad b_1 = \frac{E_{20}(\mathbf{q})}{2A}, \quad b_2 = \frac{E_{01}(\mathbf{q})}{2A}
$$

$b_0 + b_1 + b_2 = 1$, $b_i \ge 0$ が内部で成り立ちます。実装では **2 パラメータ** に減らすことが多く、$u = b_1$, $v = b_2$, $b_0 = 1 - u - v$ とおきます。

---

## 6.3 重心座標の計算とその微分

### 6.3.1 重心座標の式（2D スクリーン空間）

スクリーン座標で $u = b_1$, $v = b_2$ とすると、

$$
u = \frac{E_{20}(\mathbf{q})}{2A}, \qquad v = \frac{E_{01}(\mathbf{q})}{2A}
$$

$2A = E_{12}(\mathbf{p}_0)$ は頂点位置だけの関数で、$\mathbf{q}$ に依存しません。$E_{20}$, $E_{01}$ は $\mathbf{q}$ と頂点の座標の線形関数です。  
したがって $u$, $v$ は **頂点の 2D スクリーン座標** $\mathbf{p}_0, \mathbf{p}_1, \mathbf{p}_2$ と **ピクセル位置** $\mathbf{q}$ の関数です。

### 6.3.2 重心座標の頂点位置に関する微分

逆伝播では、損失 $L$ に対する $\frac{\partial L}{\partial u}$, $\frac{\partial L}{\partial v}$ が上流から渡され、**頂点のスクリーン座標** $\mathbf{p}_k = (x_k, y_k)$ への勾配 $\frac{\partial L}{\partial x_k}$, $\frac{\partial L}{\partial y_k}$ が必要になります。連鎖律より

$$
\frac{\partial L}{\partial x_k} = \frac{\partial L}{\partial u} \frac{\partial u}{\partial x_k} + \frac{\partial L}{\partial v} \frac{\partial v}{\partial x_k}, \quad
\frac{\partial L}{\partial y_k} = \frac{\partial L}{\partial u} \frac{\partial u}{\partial y_k} + \frac{\partial L}{\partial v} \frac{\partial v}{\partial y_k}
$$

です。$u = E_{20}(\mathbf{q})/(2A)$, $v = E_{01}(\mathbf{q})/(2A)$ を各 $x_k, y_k$ で偏微分します。  
$E_{20} = (x - x_2)(y_0 - y_2) - (y - y_2)(x_0 - x_2)$ より、

$$
\frac{\partial E_{20}}{\partial x_0} = y - y_2, \quad \frac{\partial E_{20}}{\partial y_0} = -(x - x_2), \quad \frac{\partial E_{20}}{\partial x_2} = -(y - y_0), \quad \frac{\partial E_{20}}{\partial y_2} = x - x_0
$$

同様に $E_{01}$, $2A = E_{12}(\mathbf{p}_0)$ の偏微分も計算できます。$u = E_{20}/(2A)$ なので

$$
\frac{\partial u}{\partial x_k} = \frac{1}{2A} \frac{\partial E_{20}}{\partial x_k} - \frac{E_{20}}{(2A)^2} \frac{\partial (2A)}{\partial x_k}
$$

のように、商の微分則で整理できます。$v$ も同様です。  
**重要**: ここで得ているのは **スクリーン座標** $(x_k, y_k)$ に関する微分です。頂点は本来 **クリップ空間**（または 3D）で与えられるため、**クリップ空間の頂点座標** への勾配は、さらに「スクリーン座標 → NDC → クリップ座標」の変換のヤコビを掛ける必要があります。それが 6.6 で扱う「スクリーン重心座標から 3D 空間への変換勾配」の一部です。

### 6.3.3 ピクセル位置 $\mathbf{q}$ を固定したときの頂点移動

$\mathbf{q}$ はピクセル中心で固定なので、**頂点を動かしたとき** $u$, $v$ がどう変わるかが逆伝播で効きます。上で求めた $\frac{\partial u}{\partial x_k}$ などは、まさに「頂点 $k$ のスクリーン座標を少し動かしたときの $u$ の変化率」です。解析的勾配（nvdiffrast など）では、この式をそのまま実装し、そのあとクリップ空間の頂点まで勾配を伝播させます。

---

## 6.4 深度・属性の補間式

### 6.4.1 スクリーン空間での線形補間（深度の例）

3 頂点の深度を $z_0, z_1, z_2$（NDC の $z$ や、その線形変換）とし、重心座標 $(u, v)$ で **線形補間** すると、

$$
z = (1-u-v) z_0 + u z_1 + v z_2
$$

となります。これは **スクリーン空間** では正しい補間ですが、**透視投影** をした場合、3D 空間で線形に変化する属性（テクスチャ座標・法線など）をこのようにそのまま線形補間すると歪みが生じます。

### 6.4.2 透視補正の式（$1/w$ 補間）

3D 空間で線形に変化する属性 $A$（色・UV・法線の各成分など）を、透視の下で正しく補間するには、**$1/w$ と $A/w$ を線形補間** し、その比で $A$ を復元します。

頂点の $w$ を $w_0, w_1, w_2$（クリップ空間の $w_c$ をそのまま使うか、同じ比になる値）、属性を $A_0, A_1, A_2$ とすると、

$$
\frac{1}{w} = \frac{1-u-v}{w_0} + \frac{u}{w_1} + \frac{v}{w_2}, \qquad
\frac{A}{w} = \frac{(1-u-v)A_0}{w_0} + \frac{u A_1}{w_1} + \frac{v A_2}{w_2}
$$

したがって

$$
A = \frac{A/w}{1/w} = \frac{ \frac{(1-u-v)A_0}{w_0} + \frac{u A_1}{w_1} + \frac{v A_2}{w_2} }{ \frac{1-u-v}{w_0} + \frac{u}{w_1} + \frac{v}{w_2} }
$$

これが **パースペクティブ補正付き** の属性補間です。深度についても、$z/w$ と $1/w$ を線形補間して $z = (z/w)/(1/w)$ とすれば、透視に合った深度が得られます。

### 6.4.3 まとめの記法

$$
\lambda_0 = 1-u-v,\quad \lambda_1 = u,\quad \lambda_2 = v
$$

とおくと、

$$
\frac{1}{w} = \sum_{k=0}^{2} \frac{\lambda_k}{w_k}, \quad \frac{A}{w} = \sum_{k=0}^{2} \frac{\lambda_k A_k}{w_k}, \quad A = \frac{ \sum_k \lambda_k A_k / w_k }{ \sum_k \lambda_k / w_k }
$$

です。$\lambda_k$ はスクリーン空間の重心座標で、頂点の **クリップ空間の $w$** と組み合わせて 3D 空間での正しい補間を実現しています。

---

## 6.5 カスタム属性の補間（色・UV 以外の任意属性）

頂点属性は **色・UV・法線** に限りません。カスタムのスカラーやベクトル（頂点カラー、追加の UV セット、タンジェントなど）も、同じルールで補間できます。

- **透視補正が必要な属性**（3D 空間で線形に変化するもの）: 上記の $A/w$ と $1/w$ の線形補間で $A$ を復元する。
- **透視補正が不要な属性**（スクリーン空間で線形でよいもの）: $A = (1-u-v)A_0 + u A_1 + v A_2$ でよい。

どちらを使うかは属性の意味によります。通常、**色・UV・法線** は透視補正を行い、**スクリーン空間の何か**（例: 特定の 2D オーバーレイ用の係数）は線形補間で十分な場合があります。  
微分可能ラスタライザでは、カスタム属性に対しても **同じ補間式** を適用し、その式は $u$, $v$, $w_k$, $A_k$ の関数なので、これらに関する偏微分を連鎖律で組み合わせれば勾配が得られます。

---

## 6.6 パースペクティブ補正の完全な微分

逆伝播で「ピクセルでの損失」から **頂点位置・頂点属性** へ勾配を流すには、補間式の **完全な微分** が必要です。ここでは、(1) スクリーン空間の線形補間と 3D 空間での属性補間の違い、(2) $w$ による除算の逆伝播、(3) $u$, $v$, $w$ の関係と連鎖律、(4) **スクリーン空間重心座標から 3D 空間（頂点）への変換勾配** を段階的に扱います。

### 6.6.1 スクリーン空間の線形補間と 3D 空間での属性補間の違い

- **スクリーン空間**: $(x_s, y_s)$ 上では、重心座標 $(u, v)$ は **頂点の 2D 座標の線形関数** として得られる。深度 $z$ を単純に $z = (1-u-v)z_0 + u z_1 + v z_2$ とすると、透視がかかったシーンでは誤った補間になる。
- **3D 空間**: 正しくは **$1/w$** と **属性/$w$**（または $z/w$）を **線形補間** し、$A = (A/w)/(1/w)$ で復元する。この「線形に補間するのは $1/w$ と $A/w$」という点が、微分式にもそのまま現れます。

したがって、逆伝播では **$A$ ではなく $A/w$ と $1/w$ の補間** を変数と見て微分し、そこから $A$ と頂点への勾配を組み立てます。

### 6.6.2 クリップ空間の $w$ による除算の逆伝播

NDC 座標は $x_{\text{NDC}} = x_c / w_c$ のように **$w_c$ で割る** 操作から得られます。この除算の逆伝播を考えます。  
$s = x_c / w_c$ とおくと、

$$
\frac{\partial s}{\partial x_c} = \frac{1}{w_c}, \quad \frac{\partial s}{\partial w_c} = -\frac{x_c}{w_c^2}
$$

です。したがって、上流から $\frac{\partial L}{\partial s}$ が渡されたとき、

$$
\frac{\partial L}{\partial x_c} = \frac{\partial L}{\partial s} \frac{1}{w_c}, \quad \frac{\partial L}{\partial w_c} \ += \frac{\partial L}{\partial s} \left(-\frac{x_c}{w_c^2}\right)
$$

（$y_c$, $z_c$ についても同様。$w_c$ への勾配は $x_c/w_c$, $y_c/w_c$, $z_c/w_c$ の 3 つの除算から集約される。）  
クリップ空間の頂点座標 $(x_c, y_c, z_c, w_c)$ は MVP の出力なので、この勾配をさらに **MVP の backward** でモデル空間の頂点やカメラパラメータに伝播できます。

### 6.6.3 重心座標 $u$, $v$ と深度 $w$ の関係（$1/w$ 補間）および偏微分の連鎖律

補間された $1/w$ と $A/w$ は

$$
\frac{1}{w} = \sum_k \frac{\lambda_k}{w_k}, \quad \frac{A}{w} = \sum_k \frac{\lambda_k A_k}{w_k}
$$

で、$A = \frac{A/w}{1/w}$ です。$w$ は $\frac{1}{w}$ の逆数なので、

$$
w = \frac{1}{\sum_k \lambda_k / w_k}
$$

です。  
損失 $L$ が補間された属性 $A$ に依存しているとします。$A = (A/w)/(1/w)$ なので、

$$
\frac{\partial L}{\partial (A/w)} = \frac{\partial L}{\partial A} \frac{\partial A}{\partial (A/w)} = \frac{\partial L}{\partial A} \cdot \frac{1}{1/w} = \frac{\partial L}{\partial A} \, w
$$

$$
\frac{\partial L}{\partial (1/w)} = \frac{\partial L}{\partial A} \frac{\partial A}{\partial (1/w)} = \frac{\partial L}{\partial A} \cdot \left( -\frac{A/w}{(1/w)^2} \right) = -\frac{\partial L}{\partial A} \, w^2 (A/w) = -\frac{\partial L}{\partial A} \, A \, w
$$

（$A = (A/w)/(1/w)$ を $1/w$ で微分すると $- (A/w)/(1/w)^2$。）  
さらに $A/w$ と $1/w$ は $\lambda_k$, $w_k$, $A_k$ の関数なので、

$$
\frac{\partial (A/w)}{\partial \lambda_k} = \frac{A_k}{w_k}, \quad \frac{\partial (1/w)}{\partial \lambda_k} = \frac{1}{w_k}
$$

$$
\frac{\partial (A/w)}{\partial A_k} = \frac{\lambda_k}{w_k}, \quad \frac{\partial (A/w)}{\partial w_k} = -\frac{\lambda_k A_k}{w_k^2}, \quad \frac{\partial (1/w)}{\partial w_k} = -\frac{\lambda_k}{w_k^2}
$$

これらと $\lambda_0 = 1-u-v$, $\lambda_1 = u$, $\lambda_2 = v$ から、$\frac{\partial L}{\partial u}$, $\frac{\partial L}{\partial v}$ および $\frac{\partial L}{\partial w_k}$, $\frac{\partial L}{\partial A_k}$ が求まります。例えば

$$
\frac{\partial L}{\partial u} = \frac{\partial L}{\partial (A/w)} \frac{A_1}{w_1} + \frac{\partial L}{\partial (1/w)} \frac{1}{w_1} - \frac{\partial L}{\partial (A/w)} \frac{A_0}{w_0} - \frac{\partial L}{\partial (1/w)} \frac{1}{w_0}
$$

（$u$ は $\lambda_1$ を 1 増やし $\lambda_0$ を 1 減らすので、$\lambda_1$, $\lambda_0$ に関する偏微分の組み合わせ。）同様に $\frac{\partial L}{\partial v}$ も書けます。

### 6.6.4 スクリーン空間重心座標から 3D 空間重心座標への変換勾配（つまずきポイント）

**問題の本質**: 逆伝播では $\frac{\partial L}{\partial u}$, $\frac{\partial L}{\partial v}$ が上流から渡されます。しかし **頂点位置** は 3D（またはクリップ空間）で与えられており、$u$, $v$ は **スクリーン座標** 上の頂点位置の関数です。したがって、

$$
\frac{\partial L}{\partial \mathbf{p}_k^{\text{clip}}} = \frac{\partial L}{\partial u} \frac{\partial u}{\partial \mathbf{p}_k^{\text{clip}}} + \frac{\partial L}{\partial v} \frac{\partial v}{\partial \mathbf{p}_k^{\text{clip}}}
$$

を求めるには、**$u$, $v$ をクリップ空間の頂点の関数と見て微分する** 必要があります。  
**連鎖**: クリップ空間の頂点 $\mathbf{p}_k^{\text{clip}} = (x_c, y_c, z_c, w_c)_k$ → NDC → スクリーン座標 $\mathbf{p}_k^s = (x_k, y_k)$ → エッジ関数・面積 $2A$ → $u$, $v$。

1. **クリップ → スクリーン**: $x_k = f_x(x_c/w_c, \ldots)$, $y_k = f_y(y_c/w_c, \ldots)$ のような変換（スケール・オフセット含む）。各頂点 $k$ について $\frac{\partial x_k}{\partial x_c}$, $\frac{\partial x_k}{\partial w_c}$ などは 6.6.2 の除算の微分で得られる。
2. **スクリーン座標 → $u$, $v$**: 6.3.2 で求めた $\frac{\partial u}{\partial x_k}$, $\frac{\partial u}{\partial y_k}$ など。
3. **合成**: $\frac{\partial u}{\partial x_c} = \frac{\partial u}{\partial x_k} \frac{\partial x_k}{\partial x_c} + \frac{\partial u}{\partial y_k} \frac{\partial y_k}{\partial x_c}$ のように、スクリーン座標を経由する連鎖律で **クリップ空間の各成分** への勾配が得られる。

さらに、**$w$ そのもの**（補間で使う $w_k$）はクリップ空間の第 4 成分なので、$\frac{\partial L}{\partial w_k}$ はそのまま $\frac{\partial L}{\partial (w_c)_k}$ としてクリップ空間のその成分に渡します。  
**まとめ**: 「スクリーン空間の重心座標 $u$, $v$ から 3D（クリップ）空間の頂点への変換勾配」は、

- $u$, $v$ → スクリーン座標 $(x_k, y_k)$ への微分（6.3.2）
- スクリーン座標 → NDC（$x_c/w_c$ など）への微分（除算の backward）
- NDC → クリップ座標 $(x_c, y_c, z_c, w_c)$ への微分（上記除算）
- 属性補間経由の $w_k$, $A_k$ への勾配（6.6.3）

を **すべて連鎖律でつなぐ** ことで得られます。実装では、この順で backward を書いていけば、頂点位置・頂点属性の両方に勾配が届きます。  
初心者がつまずくのは、「$u$, $v$ は 2D の式で計算しているのに、なぜ 3D の頂点に勾配がいるのか」と「どこで $w$ の除算の backward を入れるか」の両方です。**$u$, $v$ は 2D 投影された頂点の位置の関数** であり、その 2D 位置は **クリップ座標の $w$ 除算** で得られる、という二段階をはっきり分けて書くと理解しやすくなります。

---

## 6.7 まとめと次章への接続

- **クリッピングと NDC**: クリップ空間の $w$ 除算で NDC になり、さらに線形写像でスクリーン座標になる。$w$ が透視補正の鍵。
- **エッジ関数**: $E_{ij}(\mathbf{q})$ で内外判定と重心座標（$u$, $v$）を計算。その頂点スクリーン座標に関する微分を 6.3 で導出。
- **深度・属性の補間**: 透視補正は $1/w$ と $A/w$ の線形補間で $A$ を復元。カスタム属性も同じルールでよい。
- **パースペクティブ補正の微分**: $w$ 除算の backward、$\lambda_k$, $w_k$, $A_k$ に関する偏微分、そして **$u$, $v$ → スクリーン座標 → クリップ座標** の連鎖で、頂点位置・属性への勾配を一通りつなげる。

次章（第 7 章）では **アンチエイリアシング** を扱い、エッジのジャギーとその緩和、および微分可能ラスタライザにおける境界の扱いを学びます。

---

*前: [第 5 章 微分可能性の障壁](../Part02/Chapter05.md) | 次: [第 7 章 アンチエイリアシング](Chapter07.md)*
