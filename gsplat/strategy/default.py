from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class DefaultStrategy(Strategy):
    """デフォルト密度化戦略：3DGS原論文に従った動的ガウシアン管理
    
    物理的意味：
    - 画像の詳細が不足している領域（高勾配）に新しいガウシアンを配置
    - 過度に大きなガウシアンは細かく分割して解像度を向上
    - 透明で不要なガウシアンは削除してメモリ効率化
    - 定期的に不透明度をリセットして学習の安定性を保つ

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    # 物理的閾値パラメータ：ガウシアンの生存・増殖・削除を制御
    prune_opa: float = 0.005        # 不透明度閾値：これ以下は「見えない」として削除
    grow_grad2d: float = 0.0002     # 2D勾配閾値：画像平面での「詳細不足」判定基準
    grow_scale3d: float = 0.01      # 3D小サイズ閾値：「点的」ガウシアンの複製基準
    grow_scale2d: float = 0.05      # 2D大サイズ閾値：「ぼやけた」ガウシアンの分割基準
    prune_scale3d: float = 0.1      # 3D大サイズ閾値：「過大」ガウシアンの削除基準
    prune_scale2d: float = 0.15     # 2D大サイズ閾値：画面上で「巨大」なガウシアンの削除基準
    
    # 時間スケジュールパラメータ：学習段階に応じた密度化制御
    refine_scale2d_stop_iter: int = 0     # 2Dスケール判定停止時期
    refine_start_iter: int = 500          # 密度化開始：初期学習安定後
    refine_stop_iter: int = 15_000        # 密度化終了：過密化防止
    reset_every: int = 3000               # 不透明度リセット間隔：学習停滞防止
    refine_every: int = 100               # 密度化実行間隔：頻繁すぎず適度に
    pause_refine_after_reset: int = 0     # リセット後の休止期間
    
    # アルゴリズム制御フラグ
    absgrad: bool = False                 # 絶対勾配使用：より積極的な増殖
    revised_opacity: bool = False         # 改良不透明度計算（実験的）
    verbose: bool = False                 # 密度化過程の詳細出力
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"  # 勾配計算対象

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """密度化戦略の状態を初期化：各ガウシアンの「成績表」を作成
        
        物理的意味：
        - 各ガウシアンがどれだけ画像改善に貢献しているかを追跡
        - 勾配累積で「詳細表現の必要性」を測定
        - 可視回数で「重要度」を評価
        - これらの統計に基づいて増殖・削除を判定

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # 初回実行時にデバイス配置するため遅延初期化
        # 物理的状態変数の説明：
        # - grad2d: 各ガウシアンの画像平面勾配ノルム累積値（詳細不足度の指標）
        # - count: 各ガウシアンの可視回数累積値（重要度の指標）
        # - radii: 各ガウシアンの半径（画像解像度で正規化、サイズの指標）
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """逆伝播前処理：勾配情報を保持して密度化判定の準備
        
        物理的意味：
        - 各ガウシアンがレンダリング結果にどの程度影響するかの勾配を記録
        - この勾配が大きい = そのガウシアンの位置調整が画質改善に重要
        - 高勾配領域は「詳細が不足」している可能性が高い
        """
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        # 2D位置パラメータの勾配を保持：密度化判定で使用
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """逆伝播後処理：勾配情報に基づく動的ガウシアン管理の実行
        
        物理的意味：
        - 計算された勾配から各ガウシアンの「成績」を評価
        - 成績が悪い（低勾配・低不透明度）ガウシアンは削除
        - 成績が良い（高勾配）ガウシアンは増殖させて詳細表現を強化
        - これにより品質向上と計算効率のバランスを動的に調整
        """
        if step >= self.refine_stop_iter:
            return  # 密度化終了期間：安定した構造を維持

        # ガウシアン成績の更新
        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # ガウシアン増殖フェーズ：詳細不足領域への新規配置
            # 物理的解釈：画像の「解像度不足」部分を補強
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli}個複製, {n_split}個分割. "
                    f"総ガウシアン数: {len(params['means'])}個"
                )

            # ガウシアン刈り込みフェーズ：不要な要素の除去
            # 物理的解釈：「見えない」「過大」なガウシアンを除去してメモリ効率化
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune}個削除. "
                    f"総ガウシアン数: {len(params['means'])}個"
                )

            # 統計情報リセット：次回判定のため累積値をクリア
            state["grad2d"].zero_()     # 勾配累積値リセット
            state["count"].zero_()      # 可視回数リセット
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()  # 半径記録リセット
            torch.cuda.empty_cache()    # GPU メモリ断片化解消

        if step % self.reset_every == 0 & step > 0:
            # 定期的不透明度リセット：学習停滞の回避
            # 物理的意味：過度に不透明になったガウシアンを半透明に戻し、
            #           他のガウシアンにも学習機会を与える
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,  # 削除閾値の2倍程度に設定
            )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """ガウシアン状態統計の更新：各ガウシアンの「成績記録」を蓄積
        
        物理的意味：
        - 勾配の大きさ：そのガウシアンが画質にどれだけ影響するか
        - 可視回数：そのガウシアンがどれだけ頻繁に使われるか
        - これらを累積して各ガウシアンの重要度を評価
        """
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # 勾配を画面空間 [-1, 1] に正規化：異なる解像度での公平な比較
        # 物理的意味：画面サイズに依存しない「相対的な重要度」を計算
        if self.absgrad:
            # 絶対勾配：方向に関係なく変化の大きさのみ評価
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            # 通常勾配：変化の方向も考慮した重要度評価
            grads = info[self.key_for_gradient].grad.clone()
        # 画面座標系での正規化：pixel単位 → 相対座標
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # 初回実行時に統計バッファを初期化
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            # 2D勾配累積バッファ：各ガウシアンの「詳細不足度」記録
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            # 可視回数累積バッファ：各ガウシアンの「使用頻度」記録
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            # 半径記録バッファ：各ガウシアンの「画面占有率」記録
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # 可視ガウシアンの統計情報を累積更新
        # 物理的解釈：「見えている」ガウシアンのみが画質に貢献するため対象とする
        if packed:
            # パックモード：効率的なスパース処理
            gs_ids = info["gaussian_ids"]  # 可視ガウシアンのID [nnz]
            radii = info["radii"].max(dim=-1).values  # 最大半径 [nnz]
        else:
            # 通常モード：全ガウシアンから可視分を抽出
            sel = (info["radii"] > 0.0).all(dim=-1)  # 可視性判定 [C, N]
            gs_ids = torch.where(sel)[1]  # 可視ガウシアンのインデックス [nnz]
            grads = grads[sel]  # 可視ガウシアンの勾配のみ [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # 可視ガウシアンの半径 [nnz]
        
        # 統計の累積更新
        # 勾配ノルム累積：「詳細不足度」の蓄積
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        # 可視回数累積：「重要度」の蓄積
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # 最大半径記録：画面占有率の追跡（scatter_maxが理想的）
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # 半径を [0, 1] 画面空間に正規化
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        """ガウシアン増殖処理：詳細不足領域への新規ガウシアン配置
        
        物理的戦略：
        1. 小さいガウシアン + 高勾配 → 複製（同じ場所により密度を追加）
        2. 大きいガウシアン + 高勾配 → 分割（細かい構造を表現可能に）
        
        これにより画像の「ぼやけた」部分や「粗い」部分を段階的に改善
        """
        count = state["count"]
        # 平均勾配を計算：累積勾配 ÷ 可視回数 = 「平均的な重要度」
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        # 複製判定：小さくて重要なガウシアンを見つける
        # 物理的意味：「点的な光源」で詳細が不足している場所
        is_grad_high = grads > self.grow_grad2d  # 高勾配 = 詳細不足
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )  # 小サイズ = 点的構造
        is_dupli = is_grad_high & is_small  # 小さくて重要 → 複製候補
        n_dupli = is_dupli.sum().item()

        # 分割判定：大きくて重要なガウシアンを見つける
        # 物理的意味：「ぼやけた領域」で詳細が不足している場所
        is_large = ~is_small  # 大サイズ = 拡散的構造
        is_split = is_grad_high & is_large  # 大きくて重要 → 分割候補
        if step < self.refine_scale2d_stop_iter:
            # 画面占有率での追加分割判定：あまりに大きすぎる場合
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        # 段階的増殖実行：複製 → 分割の順序が重要
        # 1. 複製実行：小さなガウシアンを同位置にコピー
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # 2. 分割マスク調整：新規複製されたガウシアンは分割対象外
        # 理由：新しく作られたばかりのガウシアンをすぐに分割すると不安定
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # 3. 分割実行：大きなガウシアンを2つの小さなガウシアンに分解
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """ガウシアン刈り込み処理：不要なガウシアンの除去
        
        物理的基準：
        1. 低不透明度 → 「見えない」ガウシアンは削除
        2. 過大サイズ → 「ぼやけすぎ」のガウシアンは削除
        
        メモリ効率と計算速度の向上を図る
        """
        # 基本削除条件：透明すぎるガウシアンは「見えない」ので削除
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        
        if step > self.reset_every:
            # 追加削除条件：過大サイズのガウシアンを削除
            # 物理的意味：あまりに大きすぎて「ぼやけ」を生み出すガウシアンを除去
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # 画面サイズベースの削除（元実装にバグがあるため通常無効）
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # 完全性のため実装するが、refine_scale2d_stop_iter=0でデフォルト無効
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        # 削除実行：条件に該当するガウシアンを除去
        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
