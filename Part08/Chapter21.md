# 第21章 参考文献とリソース

本教材の最後に、**参考文献とリソース** をまとめます。微分可能ラスタライザに関連する主要論文（21.1）、オープンソース実装へのリンク（21.2）、そして NeRF やニューラル BRDF などの関連トピック（21.3）を紹介し、さらに学ぶときの道しるべにします。

---

## 21.1 論文一覧（OpenDR, Soft Rasterizer, DIB-R, nvdiffrast 等）

### 古典的アプローチ

- **OpenDR: Differentiable Renderer**  
  M. M. Loper, M. J. Black.  
  *European Conference on Computer Vision (ECCV)*, 2014.  
  メッシュを入力とする微分可能レンダラーの先駆け。有限差分と解析的勾配の必要性を示した。本教材の第 9 章で言及。

### ソフトラスタライゼーション

- **Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning**  
  S. Liu et al.  
  *IEEE International Conference on Computer Vision (ICCV)*, 2019.  
  ピクセルごとの重みをシグモイド・ソフトマックスで連続化し、勾配が自然に流れるようにした手法。本教材の第 10 章の中心。

- **Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer**  
  W. Chen et al. (DIB-R)  
  *NeurIPS*, 2019.  
  単一画像からの 3D メッシュ推定に、微分可能レンダラ（ソフトラスタ系）を組み合わせた。広く参照される。第 4 章・第 10 章で言及。

### 解析的勾配・nvdiffrast

- **Modular Primitives for High-Performance Differentiable Rendering**  
  S. Laine, J. Hellsten, T. Karras, Y. Lehtinen, J. Aila. (nvdiffrast)  
  *ACM Transactions on Graphics (SIGGRAPH)*, 2020.  
  rasterize と interpolate の分離、解析的勾配、OpenGL/Vulkan/CUDA バックエンド。本教材の Part VI の中心。公式リポジトリ: NVIDIA/nvdiffrast。

### その他の関連論文

- **Differentiable Monte Carlo Ray Tracing through Edge Sampling**  
  T. Loubet, N. Holzschuch, W. Jakob.  
  *ACM SIGGRAPH Asia*, 2019.  
  レイトレーシングの微分可能化。ラスタライザとは別路線だが、微分可能レンダリングの一翼。

- **PyTorch3D: A Library for Deep Learning with 3D Data**  
  N. Ravi et al.  
  *arXiv*, 2020.  
  メッシュの微分可能レンダリング（ラスタライザ・ポイントクラウド）を提供。座標系や API の比較対象として本教材の第 19 章で言及。

論文の詳細（PDF・BibTeX）は、各会議・ジャーナルの公式サイトや arXiv、著者ページから入手できます。本教材で扱った「順問題と逆問題」「離散の連続化」「解析的勾配」が、各論文でどう定式化されているかを対比すると理解が深まります。

---

## 21.2 オープンソース実装へのリンク

以下は、本教材の内容と直接関連するオープンソース実装の例です。URL は変更されることがあるため、検索で「リポジトリ名 + differentiable rasterizer」などで確認してください。

### 微分可能ラスタライザ

- **nvdiffrast**  
  NVIDIA による高性能な微分可能ラスタライザ。OpenGL / Vulkan / CUDA バックエンド。  
  GitHub: `NVIDIA/nvdiffrast`  
  本教材の Part VI で解剖した設計の参照実装。

- **PyTorch3D**  
  Facebook Research。メッシュのラスタライザ、ポイントクラウドレンダラ、損失関数など。  
  GitHub: `facebookresearch/pytorch3d`  
  座標系・API の比較や、自作との検証に有用（第 19 章 19.8）。

- **Soft Rasterizer (実装)**  
  Soft Rasterizer 論文の公式または非公式実装が GitHub に複数ある。検索: "Soft Rasterizer" "differentiable"。

- **DIB-R**  
  DIB-R 論文の公式実装。検索: "DIB-R" "Learning to Predict 3D Objects"。

### その他

- **NeRF 系**  
  ニューラルレンダリング（ボリューム）の実装。本教材ではラスタライザと対比して紹介（第 4 章）。  
  例: `bmild/nerf`, `yenchenlin/nerf-pytorch` など。

- **PyTorch 公式: Custom C++ and CUDA Extensions**  
  C++/CUDA 拡張の書き方。本教材の第 15 章 15.6 で参照。

利用する際は、各リポジトリのライセンス（MIT, BSD, など）と引用条件を確認してください。本教材の演習（第 20 章）では、まず自作で最小構成を書き、必要に応じてこれらの実装と出力・勾配を比較することを推奨します。

---

## 21.3 関連トピック（NeRF, ニューラル BRDF, 物理ベースレンダリング）

### NeRF とニューラルレンダリング

- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**  
  B. Mildenhall et al.  
  *ECCV*, 2020.  
  ボリュームとしてシーンを表現し、レイに沿って色・密度を積分する。本教材の第 4 章で、ラスタライザ（メッシュ）との違いとして紹介。編集可能なメッシュを直接は出さないが、見た目の再構成では強力。

- **派生**: Instant-NGP、3D Gaussian Splatting など、速度や表現の拡張が多数ある。ラスタライザとハイブリッドにする研究もある。

### ニューラル BRDF・材質

- 反射モデル（BRDF）をニューラルネットワークで表現し、微分可能レンダリングと組み合わせて材質を推定する研究。本教材の第 19 章 19.4（材質推定）の延長。

### 物理ベースレンダリング（PBR）

- **PBR** では、エネルギー保存や BRDF の物理モデルに基づいてシェーディングを行う。微分可能ラスタライザの「描画」部分を PBR にすると、逆レンダリングでより現実的な材質・照明の推定が可能。本教材ではシェーディングの基礎（第 8 章）までを扱い、PBR の詳細は専門書（例: *Physically Based Rendering*）に譲る。

### その他

- **3D 再構成・スキャン**: マルチビューステレオ、SfM、メッシュ復元。微分可能ラスタライザは「復元したメッシュを画像に写して損失を取る」段階で使われる。
- **テクスチャ最適化・バケツ**: 既存の 3D スキャンや UV 展開と組み合わせ、写真からテクスチャを推定する産業応用。本教材の第 19 章 19.1 の応用。

これらを押さえると、微分可能ラスタライザが「3D ビジョン・逆レンダリング」のどこに位置するかがより明確になります。

---

## 21.4 本教材の振り返りと到達目標

本教材は、**初学者から nvdiffrast 相当のライブラリをスクラッチで自作できるレベル** までを目指す構成でした。

- **Part I–II**: 数学・CG・自動微分の前提と、なぜ微分可能ラスタライザが必要か、何が障壁か。
- **Part III**: ラスタライゼーションの数学（エッジ関数、重心座標、補間、パースペクティブ補正の微分）。
- **Part IV**: 古典的アプローチ、ソフトラスタ、解析的勾配（nvdiffrast の設計思想）。
- **Part V**: 実装（考慮事項、最小 CPU 実装、コア、GPU、本格パイプライン）。
- **Part VI**: nvdiffrast の解剖、性能と堅牢性。
- **Part VII**: 逆レンダリングと最適化の応用、座標系の一貫性。
- **Part VIII**: 段階的演習と本参考文献。

index.md の「想定読者と到達目標」の表に沿って、自分の段階に合わせて該当 Part を繰り返し読み、演習（第 20 章）で手を動かすことで、逆レンダリングと自作ラスタライザの実装力を身につけられます。不明点は、上記の論文・実装・関連トピックを手がかりにさらに調べてください。

---

*前: [第 20 章 段階的演習](Chapter20.md) | [目次に戻る](../index.md)*
