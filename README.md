# BirdResearchExtractor
Array Mic を用いて音源を強調する

## 必要なファイル
Base model
https://github.com/kojima-r/BirdResearchDBPreprocessor
https://drive.google.com/drive/folders/18_2jTJM086wZ8m_5jNTDnqV4w7LJT5VX?usp=sharing
```
best_models/
  best_model.pth
model.py
label01_mapping.tsv
```
Base model2
https://github.com/qiuqiangkong/audioset_tagging_cnn
```
audioset_tagging_cnn/
metadata/
```


## 実行
```bash
python discriminator.py
python main.py
```

