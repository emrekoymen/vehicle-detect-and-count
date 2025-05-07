VENV OLUŞTURUP AŞAĞIDAKI KOMUTLARI ÇALIŞTIR

pip3 install --upgrade pip
pip3 install torch torchvision torchaudio ultralytics opencv-python pandas cjm_byte_track shapely numpy tqdm scipy matplotlib

https://drive.google.com/drive/folders/1mChJ1lB5eE6rRdV8Mnxg4gkBjWaiLPOH

BU LINKTEN yolo/ FOLDERINDAKI DATASETI INDIR, UNZIP ET, DOĞRU PATHE KOY (dataset.yaml'da belirtmen gerekiyor)

python prepare-data.py 
BU KOMUTU RUNLA

nohup python traffic-pipeline/train-yolov8.py --data traffic-pipeline/dataset.yaml --epochs 50 --batch 32
BU KOMUTU RUNLA, NOHUP TERMINALDEN BAĞIMSIZ ÇALIŞMANI SAĞLIYOR. NOHUP.OUT DOSYASINA YAZAR LOGLARI. EĞER DÜZ ÇALIŞSIN DIYORSAN NOHUPSIZ ÇALIŞTIR.

AKLINA TAKILAN BIR ŞEY VARSA ÖNCE PROCESS.md DOSYASINI OKU. BIR ŞEY ANLAMAZSAN SOR.

canın ciğerin, emre köymen




