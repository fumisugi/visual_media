# YOLO-World Failure Case Analysis

CVPR 2024論文 **YOLO-World: Real-Time Open-Vocabulary Object Detection** の学習済みモデルをローカルで動かし、COCO 2017 validation set上で画像劣化とprompt表現によるFailure Caseを評価する。

Repository: <https://github.com/fumisugi/visual_media>

生成AI利用ログ: [doc/AI_USAGE_LOG.md](doc/AI_USAGE_LOG.md)

## 実験設計

- モデル: `yolov8s-worldv2.pt`（Failure Case・改善前後で固定）
- データ: COCO 2017 valから固定seedで抽出した、互いに重ならない100枚×5組
- 対象カテゴリ: `car`, `couch`, `airplane`, `cup`
- split: validation 30枚、test 70枚
- Failure Case A: 明るさ `1.0 / 0.5 / 0.25`、Gaussian blur `σ=0 / 2 / 4`
- Failure Case B: canonical、synonym、hypernym、写真template、文template、複数形
- 副改善: 低照度画像へのgamma correction（validationでgammaを選択）
- prompt改善: canonical-anchored CLIP text prototype
- blur改善: Laplacian varianceによるblur判定＋Wiener deconvolution
- 指標: Precision、Recall、F1、COCO mAP、AP50、AP75、サイズ別AP、AR、PR曲線、bootstrap 95%区間

Failure Caseのpromptとハイパーパラメータはvalidation 30枚だけで選び、test 70枚には
固定する。同一モデルの最終改善では、既評価の3組（計300枚）だけで方式を選び、
設定ファイルを凍結してから4組目の未使用100枚を一度だけ評価する。
prompt改善ではこの4組を開発用とし、方式凍結後に5組目の未使用100枚を一度だけ評価する。

## 主結果

### 同一smallモデルの未使用holdoutで確認できた改善

- 固定条件: `yolov8s-worldv2.pt`、canonical prompt、入力640、confidence `0.25`
- 改善: blurと判定した入力だけWiener deconvolutionを適用
- blur `σ=2/4`平均F1: `0.464 → 0.548`
- F1差: `+0.084`、paired image-bootstrap 95%区間 `[+0.047, +0.122]`
- 強いblur `σ=4`: F1 `0.339 → 0.505`、mAP `0.256 → 0.404`
- clean: F1 `0.638 → 0.634`（差 `-0.003`、許容低下0.01以内）

開発用300枚の3 foldすべてで平均blur F1が上がった後に方式を凍結した。
未使用100枚ではblur強度まで含む状態を300入力中295入力で正しく分類し、blur入力はすべて
blurとして検出した。改善は主に`σ=4`へ集中し、`σ=2`のF1は
`0.589 → 0.590`、mAPは`0.510 → 0.497`だったため、軽いblurへの一様な改善は主張しない。
### Promptぶれに対する改善

- 固定条件: `yolov8s-worldv2.pt`、入力640、confidence `0.25`
- 改善: 入力prompt埋め込み25%とcanonical class埋め込み75%の正規化加重平均
- synonym/hypernym平均F1: `0.380 → 0.574`
- F1差: `+0.194`、paired image-bootstrap 95%区間 `[+0.135, +0.254]`
- synonym F1: `0.389 → 0.583`
- hypernym F1: `0.372 → 0.566`
- synonym/hypernym平均mAP: `0.443 → 0.511`
- canonical F1: `0.605 → 0.605`（canonical入力は元の埋め込みをそのまま使用）

埋め込みの混合比は既評価400枚の4 foldだけで決め、その後に抽出した未使用100枚へ
一度だけ適用した。F1差の95%区間下限が0を上回り、4つの開発foldでもすべて改善した。
推論前にtext embeddingを補正し、1クラスにつき1つのprototypeで検出する。
入力語をどのcanonical classへ対応付けるかが既知であることを要するため、
任意の自由記述promptにそのまま使える改善とは主張しない。

### Failure Caseと初期prompt実験

- canonical baseline: Precision `0.697`、Recall `0.664`、F1 `0.680`
- canonical baseline: COCO mAP `0.547`、AP50 `0.727`、AP75 `0.539`
- naive 3-prompt NMS: F1 `0.693`、AP50 `0.736`
- validationで選ばれたcategory subset NMS: F1 `0.638`、mAP `0.502`
- gamma correction: low-light F1 `0.644 → 0.667`

validationで選ばれたsubset NMSのtest F1差は `-0.042`、paired
image-bootstrap 95%区間は `[-0.098, +0.017]`であり、validationへの過適合が
見られた。naive 3-prompt NMSはtest F1差 `+0.013`、95%区間
`[-0.021, +0.047]`だった。小幅な改善可能性はあるが、
boxを統合する方式については「prompt改善が安定して有効」とは結論しない。
最終改善では、box統合ではなく推論前のtext prototype補正へ切り替えた。

## セットアップ

```bash
conda env create -f environment.yml
conda activate visual_media_yoloworld
python -m pip install -e .
```

GPU確認:

```bash
yoloworld-experiment check-gpu
```

COCO 2017 val取得:

```bash
bash scripts/download_coco.sh
```

## 実行

```bash
yoloworld-experiment prepare
yoloworld-experiment pilot
yoloworld-experiment full
yoloworld-experiment prompt-study
yoloworld-experiment extended-metrics
yoloworld-experiment visualize
yoloworld-experiment report
yoloworld-experiment artifact
yoloworld-experiment latex-report
yoloworld-experiment validate
```

または一括実行:

```bash
yoloworld-experiment all
```

pilotが継続基準を満たさない場合、一括実行は終了コード2で停止する。

同一モデル改善の探索と最終holdout評価:

```bash
yoloworld-experiment same-model-blur-screen
yoloworld-experiment same-model-blur-dev
yoloworld-experiment prepare-same-model-holdout
yoloworld-experiment same-model-blur-final
yoloworld-experiment same-model-blur-visualize
yoloworld-experiment prompt-prototype-screen
yoloworld-experiment prompt-prototype-dev
yoloworld-experiment prepare-prompt-prototype-holdout
yoloworld-experiment prompt-prototype-final
yoloworld-experiment prompt-prototype-visualize
yoloworld-experiment latex-report
yoloworld-experiment validate
```

`same-model-blur-dev`は既評価の300枚だけで方式を選ぶ。
`prepare-same-model-holdout`は凍結設定の存在とsmallモデル固定を確認してから、
残っているairplane画像数に合わせて事前quota `28/28/16/28`で未使用100枚を作る。
`same-model-blur-final`は一度だけ実行でき、結果が存在すると再評価を拒否する。
同様に`prompt-prototype-dev`は既評価400枚だけで混合比を選び、
`prepare-prompt-prototype-holdout`はそれらと重ならない100枚を作る。
`prompt-prototype-final`も結果が存在すると再評価を拒否する。

## 出力

`results/yoloworld/` に以下を生成する。

- `pilot_summary.csv`, `pilot_decision.json`
- `summary_metrics.csv`, `per_image_metrics.csv`
- `ensemble_tuning.csv`, `selected_parameters.json`
- `gamma_tuning.csv`
- `prompt_predictions.json`, `prompt_prediction_metadata.json`
- `prompt_variant_metrics.csv`, `prompt_variant_by_category.csv`
- `prompt_strategy_metrics.csv`, `prompt_strategy_by_category.csv`
- `prompt_strategy_tuning.csv`, `prompt_selected_parameters.json`
- `detection_ap_metrics.csv`, `detection_ap_by_category.csv`
- `detection_pr_curves.csv`
- `bootstrap_intervals.csv`, `bootstrap_deltas_vs_baseline.csv`
- `figures/*.png`
- `examples/*.png`
- `REPORT_DRAFT.md`
- `artifact.json`（任意の静的HTMLプレビュー用入力）
- `latex_build/`（LuaLaTeXの中間生成物）
- `YOLO_World_Failure_Case_Report.pdf`（提出用LaTeXレポート）
- `VALIDATION_REPORT.md`, `validation_results.json`
- `same_model_improvement/same_model_screening.csv`
- `same_model_improvement/same_model_development_metrics.csv`
- `same_model_improvement/selected_same_model_method.json`
- `same_model_improvement/same_model_final_summary.csv`
- `same_model_improvement/same_model_final_per_image.csv`
- `same_model_improvement/same_model_final_bootstrap.csv`
- `same_model_improvement/same_model_final_blur_detector.csv`
- `same_model_improvement/same_model_final_verdict.json`
- `figures/same_model_blur_final_holdout.png`
- `examples/same_model_blur_baseline.png`, `examples/same_model_blur_improvement.png`
- `prompt_prototype_improvement/prompt_prototype_screening.csv`
- `prompt_prototype_improvement/prompt_prototype_development_metrics.csv`
- `prompt_prototype_improvement/selected_prompt_prototype_method.json`
- `prompt_prototype_improvement/prompt_prototype_final_summary.csv`
- `prompt_prototype_improvement/prompt_prototype_final_per_image.csv`
- `prompt_prototype_improvement/prompt_prototype_final_bootstrap.csv`
- `prompt_prototype_improvement/prompt_prototype_final_verdict.json`
- `figures/prompt_prototype_final_holdout.png`
- `examples/prompt_prototype_baseline.png`, `examples/prompt_prototype_improvement.png`

`validate` は、100枚のsplit、カテゴリquota、画像別CSVと集計CSVの一致、
Precision/Recall/F1の再計算、validationで選択したパラメータの固定、
COCO AP/ARの範囲、bootstrap出力、Failure Caseに加え、5つのmanifestの相互非重複、
blur改善とprompt改善の最終holdout集計の再計算、凍結設定とmanifestのhash、
モデル不変更、改善判定を機械的に検証する。

提出用レポートの正規ソースは
`report/YOLO_World_Failure_Case_Report.tex` である。表中の実験値は、
`latex-report` 実行時に結果CSV/JSONから
`report/generated_metrics.tex` へ再生成される。LuaLaTeX版は次でビルドする。

```bash
bash scripts/build_latex_report.sh
```

この環境では既存の `latexmk` とLuaLaTeXを使用し、日本語フォントは
Noto CJKを指定している。現在の提出用PDFはA4・8ページである。
提出者情報はTeX冒頭に記載している。

静的HTML版が必要な場合だけ、別途Data Analytics pluginのportable report
builderとローカルChromeを用いる。

```bash
export DATA_ANALYTICS_PLUGIN_ROOT=/path/to/data-analytics/plugin
bash scripts/build_report.sh
```

COCO本体、モデル重み、生成結果はGit管理対象外である。

## 評価定義

confidence threshold以上の予測をscore降順に処理し、同じカテゴリかつIoU 0.50以上の未対応GT boxへgreedy matchingする。一つのGTに複数の予測が対応した場合、最初の一つだけをTP、残りをFPとする。

F1は固定またはvalidationで選んだ動作thresholdで算出する。AP/ARは
`pycocotools`のCOCO評価器を使い、raw confidence `0.01`から残したscore rankingで
算出する。prompt戦略のAPでは、validationで選んだ語・NMS・fusion方式は固定するが、
評価用のconfidence cutoffだけをraw値まで下げる。small/medium/largeはCOCOのarea区分を使う。
信頼区間は画像を単位として2,000回再標本化する。同一モデル改善の主判定は、
blur `σ=2/4`の条件別F1を各画像で平均したpaired差を使う。
prompt改善の主判定は、synonym/hypernymの条件別F1を各画像で平均したpaired差を使う。

## 注意

- 本実験はFailure Caseの比較を目的とし、YOLO-World論文の公式benchmark再現ではない。
- Ultralyticsへ移植されたYOLO-World重みを使用し、推論は外部APIではなくローカルGPU上で行う。
- 公開時にはYOLO-World、Ultralytics、COCOそれぞれのライセンスと引用条件を確認すること。
