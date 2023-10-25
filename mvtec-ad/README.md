## MVTec AD

1. Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) into `./data`
2. Extract MVTec AD features
   
   ```
   cd data_loader
   python extract_embedding.py
   ```
3. run training and testing together for all data classes
   
   ```
   python main.py
   ```