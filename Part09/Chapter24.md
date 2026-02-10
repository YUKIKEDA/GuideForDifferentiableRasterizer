# 第24章 GPU フルスクラッチ実装（DirectX 12 + C++）

Part IX のフルスクラッチ実装のうち、**GPU 側** を DirectX 12 のコンピュートシェーダーと C++ ホストで実装します。第 22 章のバッファ・デスクリプタ・I/F に従い、**D3D12 環境構築**（24.1）、**コンピュートシェーダー設計**（24.2）、**ラスタライズカーネルとアトミック深度**（24.3）、**勾配用コンピュートシェーダー**（24.4）、**ビルドと実行・検証**（24.5）を扱います。数式は第 6 章・第 23 章と同一です。

---

## 24.1 DirectX 12 環境構築（デバイス・コマンドキュー・コンピュートパイプラインステート）

### 24.1.1 初期化の流れ

ホスト側（C++）で、次の順に D3D12 リソースを用意します。

1. **デバイスとアダプタ**: `D3D12CreateDevice` でハードウェアアダプタを指定し、`ID3D12Device` を取得する。`IDXGIFactory4::EnumAdaptersByGpreference` でアダプタを列挙し、最初のアダプタやユーザー指定のものを渡す。
2. **コマンドキュー**: `ID3D12Device::CreateCommandQueue` で `D3D12_COMMAND_LIST_TYPE_DIRECT` のキューを作成する。コンピュートのみでも DIRECT でよい。
3. **コマンドアロケータとコマンドリスト**: `CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)` と `CreateCommandList(..., D3D12_COMMAND_LIST_TYPE_DIRECT)` で、フレームごとに記録・実行・リセットするコマンドリストを用意する。
4. **ルートシグネチャ**: 第 22 章 22.4.3 のデスクリプタテーブル（CBV/SRV/UAV を 1 テーブルで参照）を定義する。`D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE` で、デスクリプタの範囲（レジスタ b0～, t0～, u0～）を指定する。ラスタライズ用と勾配用で **2 種類のルートシグネチャ** を用意するか、同じレジスタレイアウトで 1 本にまとめる。
5. **デスクリプタヒープ**: `D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV` で、ラスタライズ用に CBV×1 + SRV×2 + UAV×2、勾配用に CBV×1 + SRV×4 + UAV×1 を並べたヒープを `CreateDescriptorHeap` で作成する。各バッファに対して `CreateShaderResourceView` / `CreateUnorderedAccessView` / `CreateConstantBufferView` でデスクリプタを書き込む。
6. **パイプラインステートオブジェクト（PSO）**: コンピュートシェーダーのバイナリ（DXIL）を `CreateComputePipelineState` に渡し、**ラスタライズ用 PSO** と **勾配用 PSO** の 2 つを作成する。シェーダーは 24.5 で DXC によりビルド時にコンパイルする。

### 24.1.2 バッファの作成（ヒープとリソース）

第 22 章 22.4.1–22.4.2 の通り、次のヒープ種別でリソースを作成する。

- **UPLOAD**: 頂点バッファ、三角形インデックスバッファ、定数バッファ。`D3D12_HEAP_TYPE_UPLOAD`、`D3D12_RESOURCE_STATE_GENERIC_READ`。`CreateCommittedResource` で作成し、ホストから `Map` で書き込む。
- **DEFAULT**: 幾何バッファ、フレームバッファ、深度用バッファ（24.3）、勾配バッファ。`D3D12_HEAP_TYPE_DEFAULT`。ラスタライズ前に `D3D12_RESOURCE_STATE_UNORDERED_ACCESS` にし、勾配パスで幾何・フレームを読むときは `D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE` にリソースバリアを入れる。
- **READBACK**: 勾配をホストに読み戻す用のコピー先。`D3D12_HEAP_TYPE_READBACK`。`CopyResource` の転送先とする。

定数バッファは 256 バイトアラインとする。頂点・三角形・幾何・フレーム・勾配の要素レイアウトは第 22 章 22.3.6 に従う。

### 24.1.3 1 フレームのコマンド記録

`rasterizer_d3d12_forward` の内部では、おおよそ次の順でコマンドを積む。

1. 頂点・三角形・定数を UPLOAD に `Map` で書き、`Unmap`。
2. `Reset` でコマンドリストをリセットし、`SetPipelineState(ラスタライズ PSO)`、`SetComputeRootSignature`、`SetDescriptorHeaps`、`SetComputeRootDescriptorTable` でデスクリプタをバインドする。
3. 幾何・フレーム・深度バッファを UAV として書き込むため、リソースステートを UAV にしておく。`Dispatch(グループ数X, グループ数Y, 1)` でラスタライズコンピュートを実行する。
4. 必要なら「幾何・フレームを SRV で読む」ためにリソースバリア（UAV → SRV）を入れ、補間用の別ディスパッチを行う（ラスタライズと補間を 1 パスにまとめる場合は不要）。
5. `Close` して `ExecuteCommandLists`。フェンスを `Signal` し、ホストで `Wait` してから結果を読み戻すか、次のフレームに進む。

---

## 24.2 コンピュートシェーダー設計（HLSL → DXIL、三角形並列 vs ピクセル並列）

### 24.2.1 HLSL と DXC

シェーダーは **HLSL** で記述し、**DXC**（DirectX Shader Compiler）で **コンピュートシェーダー** 用にコンパイルする。出力は **DXIL**（.dxil またはバイナリブロブ）。vcpkg の `directx-dxc` により、ビルド時に `dxc.exe` を呼び出す（24.5）。

- エントリポイント: `[shader("compute")]` または `#pragma kernel main` は不要で、エントリ名を `main` にし、`-E main` で指定する。
- スレッドグループ: `[numthreads(NX, NY, 1)]` で 1 グループあたりのスレッド数を決める。`SV_DispatchThreadID` でグローバルなスレッド ID を取得する。

### 24.2.2 三角形並列 vs ピクセル並列

**ラスタライズ** では、次の二つの設計が考えられる。

- **三角形並列**: 1 スレッド（または 1 グループ）が 1 三角形を担当する。その三角形の AABB 内のピクセルをループで走査し、エッジ関数で内外判定・重心座標・深度を計算して UAV に書き込む。複数三角形が同一ピクセルに書く場合は **深度バッファ** でアトミックに「手前」を残す（24.3）。実装が直感的で、第 23 章の CPU 実装と対応しやすい。
- **ピクセル並列**: 1 スレッドが 1 ピクセルを担当する。そのピクセルを覆盖する三角形を「全三角形を走査」して探すか、階層構造で検索する。全三角形走査は計算量が大きいため、三角形並列の方が一般的である。

本教材では **ラスタライズは三角形並列** とする。`Dispatch` のグループ数は「三角形数」をグループサイズで割った値とする（例: 1 グループ 64 スレッドなら、`(num_triangles + 63) / 64` グループ）。

**勾配** では、1 ピクセルが 1 つの「見えている三角形」に寄与し、その三角形の 3 頂点に勾配を流す。したがって **勾配パスはピクセル並列** が自然である。各スレッドが 1 ピクセルを担当し、幾何バッファから `tri_id`, `u`, `v` を読み、勾配式を計算して頂点勾配バッファに **加算** する。複数ピクセルが同一頂点に加算するため、頂点勾配の書き込みには **アトミック加算** またはリダクションが必要になる（24.4）。

---

## 24.3 ラスタライズカーネル（エッジ関数・重心座標・深度）とアトミック深度ソート

### 24.3.1 定数・バッファのバインド

第 22 章 22.4.3 のラスタライズ PSO のスロットに従う。

- **b0**: 定数バッファ。`float4x4 model_view_proj`、`uint2 resolution`、`uint num_vertices`、`uint num_triangles`。
- **t0**: 頂点バッファ（StructuredBuffer&lt;Vertex&gt;）。
- **t1**: 三角形インデックス（StructuredBuffer&lt;uint&gt;）。
- **u0**: 幾何バッファ（RWStructuredBuffer&lt;GeometryPixel&gt;）。
- **u1**: フレームバッファ（RWStructuredBuffer&lt;FramePixel&gt;）。

深度テスト用に、**深度バッファ** を追加で UAV にしておく。深度のみを格納し、アトミックで「小さい（手前）を残す」ために `RWStructuredBuffer<uint>` とする（深度を `asuint(z_over_w)` で uint 化し、`InterlockedMin` で比較する。正規化深度が 0～1 で手前が小さいなら、そのまま uint 比較でよい）。

### 24.3.2 頂点のクリップ・スクリーン座標

各頂点のモデル空間位置に `model_view_proj` を掛けてクリップ座標を得る。NDC は $x_c/w_c$, $y_c/w_c$, $z_c/w_c$。スクリーン座標は第 22 章の約束（Y 上向き、ピクセル中心が整数+0.5）に合わせて線形写像する。$w_c \le 0$ の頂点はカメラの後ろなので、その三角形はスキップする。

### 24.3.3 エッジ関数・重心座標・深度

三角形 $t$ について、3 頂点のスクリーン座標 $(x_0,y_0)$, $(x_1,y_1)$, $(x_2,y_2)$ とクリップの $z$, $w$ を取得する。$2A = E_{12}(\mathbf{p}_0)$ を計算し、$|2A| < \varepsilon$ なら退化のためスキップ。$2A < 0$ なら背面カリング（オプション）。

AABB を計算し、解像度内にクリップした範囲で、ピクセル中心 $\mathbf{q} = (i+0.5, j+0.5)$ について $E_{01}$, $E_{12}$, $E_{20}$ を計算。いずれも $\ge 0$ なら内側とし、$u = E_{20}/(2A)$, $v = E_{01}/(2A)$、深度 $z/w = (1-u-v)z_0/w_0 + u\,z_1/w_1 + v\,z_2/w_2$ を求める。

### 24.3.4 アトミック深度テストと幾何・フレームの書き込み

同一ピクセルに複数三角形が書くため、**深度バッファ** に対して `InterlockedMin(depth_buffer[pixel_idx], asuint(depth))` を実行する。戻り値は「更新前の深度」なので、**今回の深度の方が小さい（手前）ときだけ** 幾何バッファとフレームバッファを書き換える。比較は `depth < asfloat(old_depth)` で行い、勝った場合のみ `geometry_buffer[pixel_idx] = { u, v, depth, tri_id }` を書き、続けて補間した色を `frame_buffer[pixel_idx]` に書く。

- 初期値: 深度バッファは「奥」を表す値（例: `asuint(1.0f)`）でクリアしておく。幾何バッファの `tri_id` を 0（背景）でクリアする。
- 補間: 第 6 章の $1/w$ 補間で色を計算する。頂点の $w$ はクリップ空間の $w_c$、色は頂点属性の RGB を使う。

### 24.3.5 HLSL 疑似コード（ラスタライズの核）

```hlsl
// ラスタライズ: 三角形並列。SV_GroupThreadID / SV_DispatchThreadID で三角形 ID を特定
[numthreads(64, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
    uint tri_id = dtid.x;
    if (tri_id >= num_triangles) return;
    uint i0 = indices[3 * tri_id], i1 = indices[3 * tri_id + 1], i2 = indices[3 * tri_id + 2];
    // 頂点取得 → MVP → クリップ → NDC → スクリーン座標、2A, 退化・背面チェック
    // AABB をクリップしてループ: for (uint j = y_min; j <= y_max; ++j) for (uint i = x_min; i <= x_max; ++i)
    //   E01,E12,E20, u,v, depth 計算
    //   idx = j * resolution.x + i
    //   uint old_d = InterlockedMin(depth_buffer[idx], asuint(depth));
    //   if (depth < asfloat(old_d)) { geometry_buffer[idx] = ...; frame_buffer[idx] = ...; }
}
```

実際のループ範囲や定数は `resolution` などから計算する。1 三角形が多数ピクセルを覆盖する場合は、スレッド 1 本で AABB 全体をループすると負荷が偏るため、三角形をタイルに分割するなどの最適化は第 24 章の範囲外とする。

---

## 24.4 勾配用コンピュートシェーダー（逆伝播の式の実装）

### 24.4.1 バインド（勾配 PSO）

第 22 章 22.4.3 の勾配 PSO のとおり。

- **b0**: 定数バッファ（同様）。
- **t0**: 頂点 SRV、**t1**: 三角形 SRV、**t2**: 幾何バッファ SRV、**t3**: フレームバッファ SRV（勾配入力 `dL_d_frame` を別バッファで渡すなら、それを t3 にバインドする）。
- **u0**: 勾配バッファ（RWStructuredBuffer&lt;VertexGrad&gt;）。ここに頂点ごとの勾配を **加算** する。

`dL_d_frame` はフレームバッファと同じレイアウト（第 22 章 22.3.6）で、ピクセル $(i,j)$ の勾配が連続 4 float（R,G,B,alpha）で格納されているものとする。

### 24.4.2 ピクセル並列での勾配計算

各スレッドが 1 ピクセルを担当する。`Dispatch` のスレッド数は `width * height` とする（例: `numthreads(8,8,1)` でグループ数は `(width+7)/8`, `(height+7)/8`）。

1. ピクセル線形インデックスから `(i, j)` を復元し、幾何バッファから `tri_id`, `u`, `v` を読む。`tri_id == 0` なら背景のためスキップ。
2. その三角形の 3 頂点のクリップ座標・属性・$w$ を取得する。
3. `dL_d_frame` からこのピクセルの $\partial L/\partial R$, $\partial L/\partial G$, $\partial L/\partial B$（と alpha）を読む。
4. 第 6 章 6.6.3 の補間の逆伝播で $\partial L/\partial u$, $\partial L/\partial v$ および頂点属性への勾配を計算する。
5. 第 6 章 6.3.2 の式で $\partial L/\partial u$, $\partial L/\partial v$ からスクリーン座標 $(x_k, y_k)$ への勾配を求める。
6. NDC・クリップ空間への逆伝播（除算の backward、6.6.2）で $\partial L/\partial (x_c, y_c, z_c, w_c)_k$ を求める。
7. MVP の転置でモデル空間の頂点位置への勾配に変換する。
8. 得られた頂点ごとの勾配を **勾配バッファ** に **加算** する。

### 24.4.3 頂点勾配の加算（アトミック加算の制約）

複数ピクセルが同一頂点に勾配を流すため、`vertex_grad[k] += ...` は **競合** する。D3D12 の HLSL には **float の InterlockedAdd** がない。取り得る方針は次のとおりである。

- **整数エンコード**: 勾配を一定倍して整数に丸め、`InterlockedAdd` で加算する。最後にホスト側でスケールして復元する。実装は簡単だが、精度とオーバーフローに注意する。
- **リダクションパス**: 勾配パスでは「ピクセル → 頂点」の寄与を、ピクセルごとに別バッファ（例: ピクセル数 × 最大 3 頂点分の勾配）に書き、そのあと別のコンピュートパスで頂点 ID ごとにリダクション（加算）する。メモリとパス数が増えるが、float のまま扱える。
- **1 スレッド 1 頂点**: 各頂点について、その頂点を含む三角形がカバーする全ピクセルを走査して勾配を合計する。実装は重く、ピクセル数が多いと非効率である。

**推奨**: まずは **整数エンコード**（例: スケール $10^6$ で int に変換して `InterlockedAdd`）で実装し、精度が足りなければ **リダクションパス** に切り替える。勾配バッファの構造体 `VertexGrad` の各 float 成分を int にエンコードしたバッファを別途用意し、加算後に 1 パスで float に戻して `VertexGrad` に書き写す方法でもよい。

---

## 24.5 ビルド（Windows SDK、DXC）、Windows での実行・検証

### 24.5.1 HLSL のオフラインコンパイル

DXC で HLSL をコンピュートシェーダーとしてコンパイルする。vcpkg の `directx-dxc` により `dxc` が利用できる。

```bash
dxc -T cs_6_0 -E main -Fo shaders/rasterize.cso shaders/rasterize.hlsl
dxc -T cs_6_0 -E main -Fo shaders/gradient.cso shaders/gradient.hlsl
```

- `-T cs_6_0`: コンピュートシェーダー 6.0。
- `-E main`: エントリポイント名。
- `-Fo`: 出力ファイル（.cso はバイナリブロブ）。実行時にこのファイルを読み、`ID3D12Device::CreateComputePipelineState` に渡す。

### 24.5.2 CMake での DXC 呼び出し

ビルド時に上記コマンドを実行するには、`add_custom_command` で HLSL を入力、.cso を出力にし、実行ファイルをその出力に依存させる。

```cmake
find_package(directx-dxc CONFIG REQUIRED)
get_target_property(DXC_EXE directx-dxc::dxc-compiler IMPORTED_LOCATION)
# または get_filename_component(DXC_EXE "${DIRECTX_DXC_TOOL}" ABSOLUTE)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/rasterize.cso
  COMMAND ${DXC_EXE} -T cs_6_0 -E main
    -Fo ${CMAKE_CURRENT_BINARY_DIR}/shaders/rasterize.cso
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rasterize.hlsl
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rasterize.hlsl
  COMMENT "Compiling rasterize.hlsl"
)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/gradient.cso
  COMMAND ${DXC_EXE} -T cs_6_0 -E main
    -Fo ${CMAKE_CURRENT_BINARY_DIR}/shaders/gradient.cso
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gradient.hlsl
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gradient.hlsl
  COMMENT "Compiling gradient.hlsl"
)
add_custom_target(shaders ALL DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/shaders/rasterize.cso
  ${CMAKE_CURRENT_BINARY_DIR}/shaders/gradient.cso
)
add_dependencies(diff_rasterizer_main shaders)
```

実行時には、実行ファイルのパスから相対で `shaders/*.cso` を探すか、ビルドディレクトリの `shaders/` をカレントにして起動する。

### 24.5.3 Windows での実行と検証

- **実行**: アプリケーションは `rasterizer_d3d12_create(width, height)` でラスタライザを生成し、`rasterizer_d3d12_forward` で頂点・三角形・MVP を渡してディスパッチする。結果は GPU 上に残すか、必要なら幾何・フレームバッファを READBACK にコピーして `Map` で読む。
- **検証**: 第 25 章で、同じ入力に対して **CPU 実装**（第 23 章）と **GPU 実装** の出力（幾何バッファ・フレームバッファ・勾配）を比較する。差分が閾値以内であることを確認する。デバッグ時は D3D12 のデバッグレイヤー（`ID3D12Debug`）を有効にすると、リソース状態の誤りなどを検出しやすい。

---

## 24.6 まとめ

- **24.1**: D3D12 のデバイス・キュー・コマンドリスト・ルートシグネチャ・デスクリプタヒープ・PSO を用意し、UPLOAD/DEFAULT/READBACK でバッファを確保する。
- **24.2**: ラスタライズは **三角形並列**、勾配は **ピクセル並列**。HLSL は DXC で cs_6_0 にコンパイルする。
- **24.3**: エッジ関数・重心座標・深度を第 6 章の式で計算し、深度バッファで `InterlockedMin` により手前を残して幾何・フレームを書き込む。
- **24.4**: 勾配は補間→$u,v$→スクリーン→クリップ→モデルの順で逆伝播し、頂点勾配に加算する。float のアトミック加算は整数エンコードまたはリダクションパスで対応する。
- **24.5**: CMake で DXC を呼び出して HLSL を .cso にコンパイルし、Windows 上で実行して第 25 章で CPU 実装と比較・検証する。

これで、第 22 章の設計に沿った GPU フルスクラッチ実装の流れが一通り揃う。
