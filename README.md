# BirdResearchExtractor
Array Mic を用いて音源を定位・分離・識別する

## 準備

### Base model (BirdResearchDBPreprocessor)

リポジトリ内に学習済みデータなどのファイルをコピーする。

https://github.com/kojima-r/BirdResearchDBPreprocessor

https://drive.google.com/drive/folders/18_2jTJM086wZ8m_5jNTDnqV4w7LJT5VX?usp=sharing

リポジトリ内に以下のファイル
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

