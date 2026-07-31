# 生成AI利用ログ

## 使用した生成AI

- OpenAI Codex

## 主な用途

- 課題要件の整理と対象論文候補の比較
- Failure Case、改善方法、評価指標の検討
- COCOデータの抽出、YOLO-World推論、評価、可視化コードの作成補助
- Precision、Recall、F1、COCO AP/AR、bootstrap信頼区間の集計補助
- 実験結果を用いた図表とLaTeXレポート草稿の作成
- テスト、数値の整合性、PDFのページ構成と表現の確認

## 自分で修正・判断した箇所

- YOLO-Worldを対象論文とし、画像劣化とprompt表現をFailure Caseとして扱う方針を決定した。
- 評価用データを手法選択に使わないよう、開発用データと評価用データを分離した。
- prompt ensembleの結果を確認し、改善が安定しなかったため主手法には採用しなかった。
- prompt補正とblur-aware Wiener前処理の結果を確認し、改善内容と主張範囲を決定した。
- 強いblurでの改善だけでなく、軽いblurやclean画像での小さな性能低下もLimitationsに記載した。
- 生成されたコードをローカル環境で実行し、テスト、集計値、図表、引用、レポート本文を確認・修正した。
