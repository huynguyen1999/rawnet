# Hướng dẫn cài đặt và chạy thực nghiệm

## Các yêu cầu của máy
- Cần có CUDA GPU, nếu máy tính không hỗ trợ thì có thể sử dụng Google Cloud Deep Learning VM.
- Cần cài đặt sẵn Docker.
- Ổ cứng cần trống 200GB để thực hiện tải và giải nén bộ dữ liệu VoxCeleb1.

## Các thông số thực nghiệm
- L2 regularization
- Optimizer: Adam
- Kích thước mini batch: 20
- Tốc độ học: 0.001
- Tham số suy giảm: 0.0001
- Center loss lambda: 0.001
- Recurrent dropout cho lớp GRU: 0.3
- Epochs: 20 (giới hạn do chi phí Google Cloud)

## Cách chạy mã nguồn
1. Giải nén source code thành thư mục RawNet
2. Chạy image nvidia cho tương xứng với môi trường huấn luyện mô hình của tác giả bài báo
- `cd RawNet && docker run --gpus all -it --rm -v ./:/code -v ./DB:/DB -v ./:/exp nvcr.io/nvidia/tensorflow:18.05-py3`
3. Trong terminal của container, thực hiện các bước sau:
- Đi vào thưc mục chứa source code: 
cd /code
- Tải các packages: 
pip install -r requirements.txt
- Thực hiện tải dataset và bộ trọng số tốt nhất của tác giả:
python ./src/01.download_dataset.py
- Tiền xử lý những đoạn âm thanh của người nói: 
python ./src/02.pre_process_waveform.py
- Thực hiện huấn luyện mô hình theo thông số bên trên: 
python ./src/03.trn_RawNet.py
- Để đánh giá mô hình tốt nhất trên tập validation vừa được huấn luyện:
python ./src/04.evaluate_model.py --weights=/code/model/networks/RawNet_pre_train_reproduce_RawNet/models_RawNet/best_model_on_validation.h5
- Để đánh giá mô hình của tác giả:
python ./src/04.evaluate_model.py --weights=/code/model/RawNet_weights.h5

