# nishikawaGradMayekawa
2021年度の西川卒論のコードを整理したもの

## Environment
環境構築は kaggle-docker を使用した．


## How To Use
1. 前処理
	1. data/rawData_df.csvに正しく録音できていたデータのパスが含まれている
	2. script/01_preprocessing/splitWave.py を実行することで
		data/ 以下に10秒ごとに分割された音響データが保存される
2. モデルの学習  
	script/02_trainAllChannel.py を実行すると各チャンネルでのモデルが学習される．
	学習結果は適宜 result に保存される．
3. 各チャンネルを組み合わせて異常判定  
	1. script/03_ensambleAllChannel.py を実行するとそれぞれの提案手法で異常判定が行われる．
		最終的なモデルの判定は result/final_prediction.csvとして保存される．
	2. script/04_visualizeAnomalyScore.py を実行すると異常スコアが可視化される．
4. 異常音の抽出再構成
	1. script/06_reconstructWave.py を実行すると異常音が再構成される．
	2. script/07_visualizeAnomalySound.py で異常音のスペクトログラムが可視化される．
5. 異常音源方向の特定
	1. script/offlineLocalization.n のネットワークを HARK で動かして音源定位を行った結果が result/sourcePosition.npy
	2. script/08_visualizeAnomalyLocation.py で異常音源方向が可視化される
