from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import ultralytics

from .config import resolve_path


def _metric_line(row: pd.Series) -> str:
    return (
        f"Precision {row['precision']:.3f}, Recall {row['recall']:.3f}, "
        f"F1 {row['f1']:.3f}"
    )


def _environment_metadata() -> dict[str, Any]:
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            if torch.cuda.is_available()
            else None
        ),
    }
    try:
        metadata["nvidia_driver"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        metadata["nvidia_driver"] = None
    return metadata


def create_report_draft(config: dict[str, Any]) -> Path:
    results_dir = resolve_path(config, config["paths"]["results_dir"])
    frame = pd.read_csv(results_dir / "summary_metrics.csv")
    test = frame[frame["scope"] == "test"].set_index("condition")
    category_frame = pd.read_csv(results_dir / "metrics_by_category.csv")
    category_test = category_frame.set_index(["condition", "category"])
    prompt_strategies = pd.read_csv(
        results_dir / "prompt_strategy_metrics.csv"
    )
    prompt_test = prompt_strategies[
        prompt_strategies["scope"] == "test"
    ].set_index("strategy")
    prompt_validation = prompt_strategies[
        prompt_strategies["scope"] == "validation"
    ].set_index("strategy")
    detection_ap = pd.read_csv(results_dir / "detection_ap_metrics.csv")
    detection_ap_test = detection_ap[
        detection_ap["scope"] == "test"
    ].set_index("condition")
    bootstrap_deltas = pd.read_csv(
        results_dir / "bootstrap_deltas_vs_baseline.csv"
    )
    f1_deltas = bootstrap_deltas[
        bootstrap_deltas["metric"] == "f1"
    ].set_index("candidate")
    with (results_dir / "selected_parameters.json").open(
        "r", encoding="utf-8"
    ) as handle:
        selected = json.load(handle)
    with (results_dir / "prompt_selected_parameters.json").open(
        "r", encoding="utf-8"
    ) as handle:
        prompt_selected = json.load(handle)
    manifest_path = resolve_path(config, config["paths"]["subset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_entries = [
        item for item in manifest["images"] if item["split"] == "validation"
    ]
    test_entries = [
        item for item in manifest["images"] if item["split"] == "test"
    ]
    total_image_count = len(manifest["images"])
    validation_image_count = len(validation_entries)
    test_image_count = len(test_entries)
    validation_gt_count = sum(
        len(item["annotations"]) for item in validation_entries
    )
    test_gt_count = sum(len(item["annotations"]) for item in test_entries)
    validation_category_gt = {
        category: sum(
            annotation["category"] == category
            for item in validation_entries
            for annotation in item["annotations"]
        )
        for category in config["categories"]
    }
    test_small_gt_count = sum(
        float(annotation["area"]) < 32**2
        for item in test_entries
        for annotation in item["annotations"]
    )
    gamma_tuning = pd.read_csv(results_dir / "gamma_tuning.csv")
    best_gamma_f1 = float(gamma_tuning["f1"].max())
    gamma_tie_count = int(
        (gamma_tuning["f1"].astype(float) - best_gamma_f1)
        .abs()
        .le(1e-12)
        .sum()
    )

    original = test.loc["original"]
    brightness = test.loc["brightness_0.25"]
    blur = test.loc["blur_sigma_4"]
    canonical = test.loc["prompt_canonical"]
    synonym = test.loc["prompt_synonym"]
    hypernym = test.loc["prompt_hypernym"]
    baseline = test.loc["baseline_tuned"]
    ensemble = test.loc["prompt_ensemble"]
    lowlight = test.loc["lowlight_baseline"]
    gamma = test.loc["gamma_correction"]
    ensemble_params = selected["ensemble"]
    gamma_value = selected["gamma_correction"]["gamma"]
    car_canonical = category_test.loc[("prompt_canonical", "car")]
    car_synonym = category_test.loc[("prompt_synonym", "car")]
    car_hypernym = category_test.loc[("prompt_hypernym", "car")]
    cup_canonical = category_test.loc[("prompt_canonical", "cup")]
    cup_synonym = category_test.loc[("prompt_synonym", "cup")]
    prompt_baseline = prompt_test.loc["canonical_baseline"]
    prompt_naive = prompt_test.loc["naive_three_prompt_nms"]
    prompt_fixed = prompt_test.loc["category_best_fixed"]
    prompt_primary = prompt_test.loc[prompt_selected["primary_strategy"]]
    baseline_ap = detection_ap_test.loc["strategy/canonical_baseline"]
    primary_ap = detection_ap_test.loc[
        f"strategy/{prompt_selected['primary_strategy']}"
    ]
    primary_delta = f1_deltas.loc[
        f"strategy/{prompt_selected['primary_strategy']}"
    ]
    fixed_prompt_summary = " / ".join(
        f"{category}: {parameters['variant']}"
        for category, parameters in prompt_selected[
            "fixed_category_selection"
        ].items()
    )
    validation_gt_range = (
        f"{min(validation_category_gt.values())}〜"
        f"{max(validation_category_gt.values())}"
    )
    gamma_selection_note = (
        f"{gamma_tie_count}候補が最高F1で同率だったため、同率候補の中から"
        "画像変形が最も弱い値をtie-breakで選んだ"
        if gamma_tie_count > 1
        else "validation F1が最大だった候補を選んだ"
    )

    environment = _environment_metadata()
    with (results_dir / "environment_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(environment, handle, indent=2, ensure_ascii=False)

    report = f"""# YOLO-WorldのFailure Case Analysisとprompt改善

## 提出者情報

- 氏名: **要記入**
- 学籍番号: **要記入**
- 所属専攻: **要記入**
- 所属研究室: **要記入**
- 研究テーマ: **要記入**

## Technical summary

YOLO-WorldをCOCO 2017 validation setから抽出した{total_image_count}枚で評価し、{validation_image_count}枚で条件を調整、{test_image_count}枚を未使用のテスト画像として評価した。画像劣化とprompt表現という異なる2種類のFailure Caseを確認し、低照度補正に加えて6種類のpromptと複数の選択・統合法を比較した。

- 通常画像の結果: {_metric_line(original)}
- 明るさ25%の結果: {_metric_line(brightness)}
- Gaussian blur σ=4の結果: {_metric_line(blur)}
- gamma correctionの結果: {_metric_line(gamma)}
- validation選択prompt戦略の結果: {_metric_line(prompt_primary)}

強いblurではRecallが{original['recall']:.3f}から{blur['recall']:.3f}へ変化した。これは誤検出数だけでなく、対象に対応するboxを出せるかに影響するFailure Caseである。prompt表現では、canonical、synonym、descriptive/hypernymのF1が{canonical['f1']:.3f}、{synonym['f1']:.3f}、{hypernym['f1']:.3f}となった。ただし全カテゴリで一様に変化したわけではない。validationで選んだ`{prompt_selected['primary_strategy']}`ではF1が{prompt_baseline['f1']:.3f}から{prompt_primary['f1']:.3f}、COCO mAPが{baseline_ap['ap']:.3f}から{primary_ap['ap']:.3f}となった。

## 対象論文と手法理解

対象論文は Cheng et al., **“YOLO-World: Real-Time Open-Vocabulary Object Detection”**, CVPR 2024 main conferenceである。通常の物体検出器が学習時に定義された固定カテゴリだけを予測するのに対し、YOLO-Worldはテキストで与えたカテゴリを検出対象として指定できる。

手法の中心は以下である。

1. YOLO系の画像特徴抽出・物体検出機構を基礎とする。
2. CLIP text encoderでカテゴリpromptをベクトル化する。
3. RepVL-PANで画像特徴とテキスト特徴を融合する。
4. region-text contrastive lossにより、画像領域と対応する語彙を近付けて学習する。
5. 推論前にpromptの埋め込みを計算する“prompt-then-detect”方式を使う。

検出器には公開学習済み重み`yolov8s-worldv2.pt`を用い、ローカルGPUで推論した。

## Image degradation changes detection performance

![Detection metrics by image degradation](figures/failure_image_corruptions.png)

入力画像と正解boxを固定し、明るさまたはblurのみを変更した。通常画像に対して、明るさ25%ではF1が {brightness['f1'] - original['f1']:+.3f}、blur σ=4では {blur['f1'] - original['f1']:+.3f} 変化した。

blur σ=4ではTPが{int(original['tp'])}件から{int(blur['tp'])}件、FNが{int(original['fn'])}件から{int(blur['fn'])}件、Precisionが{original['precision']:.3f}から{blur['precision']:.3f}へ変化した。この結果は、ぼけによって輪郭や局所textureが失われ、候補boxに十分なconfidenceを与えられなくなった可能性と整合する。ただし内部特徴の可視化を行っていないため、これは結果から導いた仮説であり、因果的に確認したものではない。

代表例:

![Failure under Gaussian blur](examples/failure_under_gaussian_blur_sigma4.png)

## Prompt wording changes open-vocabulary detection

![Detection metrics by prompt wording](figures/failure_prompt_wording.png)

画像を固定し、canonical term、synonym、descriptive/hypernymの3種類を比較した。canonicalのF1は {canonical['f1']:.3f}、synonymは {synonym['f1']:.3f}、descriptive/hypernymは {hypernym['f1']:.3f} だった。

カテゴリ別には `car` のRecallがcanonical `{car_canonical['recall']:.3f}`、synonym `automobile` `{car_synonym['recall']:.3f}`、descriptive/hypernym `motor vehicle` `{car_hypernym['recall']:.3f}`まで低下した。一方、`cup` はcanonical F1 `{cup_canonical['f1']:.3f}`に対してsynonym `mug` が `{cup_synonym['f1']:.3f}`であり、同義語が必ず悪いわけではない。

この差は、text encoderが辞書的な同義関係だけでなく、事前学習データ中の語の使われ方と画像との対応を反映するためと考えられる。特に広い上位概念や説明句は、対象領域との対応がcanonical termより曖昧になる可能性がある。

## Gamma correction improves low-light detection

![Low-light preprocessing metrics](figures/improvement_gamma_correction.png)

検証用{validation_image_count}枚だけでgammaを選び、`gamma={gamma_value:.2f}`を採用した。{gamma_selection_note}。明るさ25%画像のF1 {lowlight['f1']:.3f}に対し、補正後は {gamma['f1']:.3f}となり、{gamma['f1'] - lowlight['f1']:+.3f}変化した。Recallは {lowlight['recall']:.3f} から {gamma['recall']:.3f}へ変化した。

代表例では補正前に正解boxと一致する予測がなかったが、補正後に低confidenceながら1件のTPが得られた。同時にFPも残っており、補正が常に予測を正しくするわけではない。

| 低照度条件 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 明るさ25% | {int(lowlight['tp'])} | {int(lowlight['fp'])} | {int(lowlight['fn'])} | {lowlight['precision']:.3f} | {lowlight['recall']:.3f} | {lowlight['f1']:.3f} |
| Gamma correction | {int(gamma['tp'])} | {int(gamma['fp'])} | {int(gamma['fn'])} | {gamma['precision']:.3f} | {gamma['recall']:.3f} | {gamma['f1']:.3f} |

![Low-light baseline](examples/low_light_baseline.png)

![Gamma correction](examples/gamma_correction.png)

## Prompt ensemble was not an effective improvement

![Single-prompt and ensemble metrics](figures/improvement_prompt_ensemble.png)

検証用{validation_image_count}枚でconfidence thresholdとNMS IoUを選択した。採用値はconfidence {ensemble_params['confidence_threshold']:.2f}、NMS IoU {ensemble_params['nms_iou']:.2f}。同じthresholdで、単一promptのF1 {baseline['f1']:.3f}に対し、ensembleは {ensemble['f1']:.3f}だった。

## Prompt改善を一通り比較

![Prompt strategy comparison](figures/prompt_strategy_comparison.png)

prompt候補をcanonical、synonym、hypernym、`a photo of ...`、`... in the image`、複数形の6種類へ増やした。改善法は、(1) 3語を単純NMS統合、(2) カテゴリ別に語を選択、(3) 語とthresholdをカテゴリ別選択、(4) prompt subsetを選んでNMS、(5) validation F1を信頼度としてweighted box fusion、を比較した。語・threshold・NMS IoU・fusion係数はvalidation {validation_image_count}枚だけで決めた。

| Prompt戦略 | Validation F1 | Test Precision | Test Recall | Test F1 | Test AP50 |
|---|---:|---:|---:|---:|---:|
| Canonical baseline | {prompt_validation.loc['canonical_baseline', 'f1']:.3f} | {prompt_baseline['precision']:.3f} | {prompt_baseline['recall']:.3f} | {prompt_baseline['f1']:.3f} | {baseline_ap['ap50']:.3f} |
| Naive 3-prompt NMS | {prompt_strategies[(prompt_strategies['scope'] == 'validation') & (prompt_strategies['strategy'] == 'naive_three_prompt_nms')]['f1'].iloc[0]:.3f} | {prompt_naive['precision']:.3f} | {prompt_naive['recall']:.3f} | {prompt_naive['f1']:.3f} | {detection_ap_test.loc['strategy/naive_three_prompt_nms', 'ap50']:.3f} |
| Category word (threshold固定) | {prompt_strategies[(prompt_strategies['scope'] == 'validation') & (prompt_strategies['strategy'] == 'category_best_fixed')]['f1'].iloc[0]:.3f} | {prompt_fixed['precision']:.3f} | {prompt_fixed['recall']:.3f} | {prompt_fixed['f1']:.3f} | {detection_ap_test.loc['strategy/category_best_fixed', 'ap50']:.3f} |
| Validation-selected subset NMS | {prompt_selected['primary_validation_f1']:.3f} | {prompt_primary['precision']:.3f} | {prompt_primary['recall']:.3f} | {prompt_primary['f1']:.3f} | {primary_ap['ap50']:.3f} |

validation全体F1で選ばれた本命は`{prompt_selected['primary_strategy']}`だった。test F1は **{prompt_baseline['f1']:.3f} → {prompt_primary['f1']:.3f}**（差 {primary_delta['estimate']:+.3f}、paired image-bootstrap 95%区間 [{primary_delta['ci_low']:+.3f}, {primary_delta['ci_high']:+.3f}]）だった。mAPは **{baseline_ap['ap']:.3f} → {primary_ap['ap']:.3f}**であり、F1の動作点とconfidence ranking全体を分けて解釈する。

thresholdを0.25に固定してカテゴリごとに語だけを変える方法のtest F1は{prompt_fixed['f1']:.3f}だった。選択内容は `{fixed_prompt_summary}` である。validationで本命に選ばれた方法と異なる場合は、post-hocに本命へ変更せず参考結果として扱う。

![Canonical prompt example](examples/prompt_baseline.png)

![Category-specific prompt example](examples/category_specific_prompt.png)

![Prompt sensitivity heatmap](figures/prompt_variant_recall_heatmap.png)

カテゴリ差は大きい。testで`car`のRecallはcanonical {car_canonical['recall']:.3f}、synonym {car_synonym['recall']:.3f}、hypernym {car_hypernym['recall']:.3f}だった。一方、`cup`のF1はcanonical {cup_canonical['f1']:.3f}、synonym {cup_synonym['f1']:.3f}だった。したがって、すべてのカテゴリへ同じ言い換えを一律適用するよりカテゴリ別に扱う方が妥当である。

![Precision-recall curves](figures/prompt_pr_curves.png)

![Bootstrap intervals](figures/prompt_f1_bootstrap_intervals.png)

## Scope, data, and metric definitions

- 対象論文: *YOLO-World: Real-Time Open-Vocabulary Object Detection*（CVPR 2024）
- モデル: `yolov8s-worldv2.pt`
- データ: COCO 2017 valから固定seedで抽出した{total_image_count}枚
- validation: {validation_image_count}枚・正解box {validation_gt_count}件
- test: {test_image_count}枚・正解box {test_gt_count}件
- 対象カテゴリ: car、couch、airplane、cup
- confidence threshold: 0.25
- box matching threshold: IoU 0.50
- TP: 同一カテゴリの未対応GT boxとIoU 0.50以上で一致した予測
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: PrecisionとRecallの調和平均
- AP50 / AP75: IoU 0.50 / 0.75におけるinterpolated Precision–Recall曲線の面積
- COCO mAP: IoU 0.50から0.95まで0.05刻みで平均したAP
- AR@100: 1画像・カテゴリ当たり最大100予測での平均Recall
- AP small / medium / large: COCOのarea区分によるサイズ別AP
- 不確実性: test画像を単位とする2,000回bootstrap 95%区間

通常画像・canonical promptの標準指標は、mAP {baseline_ap['ap']:.3f}、AP50 {baseline_ap['ap50']:.3f}、AP75 {baseline_ap['ap75']:.3f}、AR@100 {baseline_ap['ar_100']:.3f}だった。APはconfidenceを0.01から残したranked predictionsで算出し、F1は固定またはvalidation選択thresholdで算出した。

## Methodology

画像劣化実験ではpromptをcanonical termに固定し、明るさ倍率とGaussian blurのみを変えた。prompt実験では元画像を固定し、4カテゴリそれぞれについて6表現を比較した。改善実験ではgamma、promptの語、confidence threshold、NMS IoU、fusion係数をvalidation {validation_image_count}枚だけで選択した。test {test_image_count}枚は最終比較までパラメータ選択に使用しなかった。

予測boxはconfidence降順に処理し、同一カテゴリでIoU 0.50以上となる未対応GTへgreedy matchingした。1つのGTへ複数boxが重なった場合、最初の予測だけをTP、残りをFPとした。データ抽出、画像変換、推論、集計、グラフ生成は設定ファイルから一括再実行できる。

## Execution environment

- OS: `{environment['platform']}`
- GPU: `{environment['gpu']}` / `{environment['gpu_memory_gib']} GiB`
- NVIDIA driver: `{environment['nvidia_driver']}`
- Python: `{environment['python']}`
- PyTorch: `{environment['torch']}`（CUDA build `{environment['torch_cuda_build']}`）
- Ultralytics: `{environment['ultralytics']}`

## Limitations and robustness

- {total_image_count}枚・4カテゴリの限定評価であり、COCO全体の性能を表すものではない。
- synonymとhypernymの品質はカテゴリごとに異なり、prompt集合の選び方に依存する。
- 画像劣化は人工的に生成しており、実環境の低照度やmotion blurを完全には再現しない。
- gamma correctionは人工的な明るさ低下に合わせた改善で、実環境のnoise増幅を評価していない。
- validationのGTは{validation_gt_count}件で、gamma候補の選択にはなお標本依存性がある。
- promptのカテゴリ別selectionに使えるvalidation GTは各カテゴリ{validation_gt_range}件である。
- testでのprompt改善差のbootstrap区間は0を含み、有意な改善を主張できない。
- testのsmall GTは{test_small_gt_count}件で、サイズ別APは標本数とともに解釈する必要がある。

## Recommended next steps

次は複数seedまたはcross-validationでprompt選択の安定性を評価する。候補語を増やす場合も、同じtestを見ながら選ばずvalidation内で完結させる。実写low-light画像でもgamma correctionを評価し、noise増幅とのtrade-offを確認する。

## Further questions

- 小物体に限定した場合、prompt wordingの影響は強くなるか。
- 低照度補正とprompt ensembleを組み合わせた場合、改善は加算的か。

## 生成AI利用

Codexを課題要件整理、候補論文の比較、実験設計、環境構築、評価コード作成、結果集計、図表作成、文章草稿に使用した。生成内容は実行結果および参照文献と照合し、実験条件、数値、考察は著者が確認した。

## References

1. T. Cheng et al., “YOLO-World: Real-Time Open-Vocabulary Object Detection,” CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html
2. Official YOLO-World implementation. https://github.com/AILab-CVC/YOLO-World
3. Ultralytics YOLO-World documentation. https://docs.ultralytics.com/models/yolo-world/
4. COCO dataset. https://cocodataset.org/
"""
    output = results_dir / "REPORT_DRAFT.md"
    output.write_text(report, encoding="utf-8")
    return output
