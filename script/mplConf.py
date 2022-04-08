import matplotlib as mpl
import matplotlib.pyplot as plt
# import japanize_matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams

### figsize
rcParams["figure.figsize"] = [6, 4.5]  # 図の縦横のサイズ([横(inch),縦(inch)])
rcParams["font.size"] = 8 # 全体のフォントサイズが変更されます。
rcParams['xtick.labelsize'] = 8 # 軸だけ変更されます。
rcParams['ytick.labelsize'] = 8 # 軸だけ変更されます
rcParams["axes.labelsize"] = 8	# 軸ラベルのフォントサイズ
rcParams["figure.subplot.left"] = 0.05  # 余白
rcParams["figure.subplot.bottom"] = 0.05# 余白
rcParams["figure.subplot.right"] =0.95 # 余白
rcParams["figure.subplot.top"] = 0.95   # 余白

### font
rcParams['font.family'] = 'sans-serif'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic',
                                'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic',
                                'VL PGothic', 'Noto Sans CJK JP']

### conf
rcParams["figure.dpi"] = 150
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["xtick.top"] = True
rcParams["xtick.bottom"] = True
rcParams["ytick.left"] = True
rcParams["ytick.right"] = True

rcParams["axes.linewidth"] = 0.3	# グラフ囲う線の太さ
rcParams["axes.grid"] = False	# グリッドを表示するかどうか

rcParams["legend.loc"] = "best"	# 凡例の位置、"best"でいい感じのところ
rcParams["legend.fontsize"] = 6