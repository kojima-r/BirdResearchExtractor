# BirdResearchExtractor
Array Mic を用いて音源を定位・分離・識別する

## 準備

### Base model (BirdResearchDBPreprocessor)

リポジトリ内に学習済みデータなどのファイルをコピーする。

https://github.com/kojima-r/BirdResearchDBPreprocessor

https://drive.google.com/drive/folders/18_2jTJM086wZ8m_5jNTDnqV4w7LJT5VX?usp=sharing

```
BirdResearchExtractor/
  - best_models/
    - best_model.pth
  - model.py
  - label01_mapping.tsv
```
### Base model2 (PANNs)

リポジトリ内に以下のPANNsのファイル・フォルダをコピーする。

https://github.com/qiuqiangkong/audioset_tagging_cnn
```
BirdResearchExtractor/
  - audioset_tagging_cnn/
  - metadata/
```

## 環境構築コマンド
このリポジトリを展開
```bash
git clone <this repository>
cd BirdResearchExtractor
```

環境構築
```bash
git clone https://github.com/kojima-r/BirdResearchDBPreprocessor

conda create -n hbb python=3.9

pip install gdown
# best_model_16k_20200105.pth
gdown "https://drive.google.com/uc?export=download&id=1COJkhaP9wf3Q5ClUN_VoTfjvf0NIPkP2"
# label01_mapping.tsv
gdown "https://drive.google.com/uc?export=download&id=12XJvNxXaGv_LHjZOelojjv1DC7W_rpnT"

mkdir best_models
cp best_model_16k_20200105.pth best_models/best_model.pth

git clone https://github.com/kojima-r/BirdResearchDBPreprocessor.git
cp BirdResearchDBPreprocessor/model.py ./

git clone https://github.com/qiuqiangkong/audioset_tagging_cnn
cp -r audioset_tagging_cnn/metadata ./

 
conda create -n hbb python=3.9
conda activate hbb

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

conda install -c conda-forge librosa
conda install pandas h5py
pip install soundfile torchlibrosa

conda install pyqtgraph
pip install PySide6
conda install pyaudio seaborn

pip install git+https://github.com/kojima-r/MicArrayX.git
pip install git+https://github.com/kojima-r/HARK_TF_Parser.git
```

Note: Python 3.10だとpyaudioのバグでエラーが発生する(https://stackoverflow.com/questions/70344884/pyaudio-write-systemerror-py-ssize-t-clean-macro-must-be-defined-for-format)


## 実行
display.py → discriminator.py → extractor.py の順に実行すると上手くいきやすい。

discriminator.py を実行してからモデルがロードされるまでしばらく待ってから extractor.py を実行する

extractor はマイク1から起動しないとオーディオストリーミングの起動時にエラーになることがある。（原因不明）

```bash
python display.py
python discriminator.py
python extractor.py 1
python extractor.py 0
```

