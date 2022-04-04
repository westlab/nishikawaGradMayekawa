# nishikawaGradMayekawa
2021年度の西川卒論のコードを整理したもの


## How To Use
1. 前処理
	1. data/rawData_df.csvに正しく録音できていたデータのパスが含まれている
	2. preprocessing/splitWave.py を実行することで
		data/ 以下に10秒ごとに分割された音響データが保存される
	3. stft.py を実行することで分割された音響データに対してSTFTをかけたものが得られる．
2. モデルの学習
3. 各チャンネルを組み合わせて異常判定
4. 異常音の抽出再構成
5. 異常音源方向の特定
