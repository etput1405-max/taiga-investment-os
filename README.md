# 大河専用 投資判断OS

GitHub Actionsが平日18:40に日本株を自動取得し、`data.json`を更新します。

## 初回設定

1. リポジトリをPublicへ変更
2. Settings → Pages
3. Deploy from a branch
4. Branchを`main`、フォルダを`/ (root)`にしてSave
5. Actions → Update investment ranking → Run workflow

## 現在の範囲

無料枠と取得安定性を優先し、初期設定は最大800銘柄です。
価格推移・押し目・相対強度・ボラティリティによる候補抽出です。
決算、適時開示、ニュースのAI解析は未実装です。
