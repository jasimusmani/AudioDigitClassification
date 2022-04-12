import cnn_attention.model as cnn_attention_model
import lstm_svm.model as lstm_model
import gdown
import zipfile
import os

if __name__ == '__main__':

    dataset_path = 'data/recordings/'

    #download th erecording waves form the googleDrive
    url = 'https://drive.google.com/file/d/1TMjMYrTKwOWjYSQw9-JqQZ8vTDCcLLz5/view?usp=sharing'
    output = 'data/recordings.zip'
    if not os.path.exists(dataset_path):
        gdown.download(url, output, quiet=False, fuzzy=True)
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('data')


    # cnn_attention_accuracy = cnn_attention_model.train_model()
    # print("cnn_attention_accuracy:", cnn_attention_accuracy)

    # lstm_accuracy = lstm_model.train_lstm_model(dataset_path)
    # print("lstm_accuracy:", lstm_accuracy)

    svm_accuracy = lstm_model.train_svm_model(dataset_path)
    print("svm_accuracy:", svm_accuracy)



