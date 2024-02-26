import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, f1_score
from some_classes import TextClassificationDataset, DistilBertForSequenceClassification

data_path = "./data"
data = pd.read_json(data_path+"/测试集.json", lines=True)
MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 256
NUM_CLASSES = 42

label_dict = {'ARTS': 0, 'ARTS & CULTURE': 1, 'BLACK VOICES': 2, 'BUSINESS': 3, 'COLLEGE': 4, 'COMEDY': 5, 'CRIME': 6, 'CULTURE & ARTS': 7, 'DIVORCE': 8, 'EDUCATION': 9, 'ENTERTAINMENT': 10, 'ENVIRONMENT': 11, 'FIFTY': 12, 'FOOD & DRINK': 13, 'GOOD NEWS': 14, 'GREEN': 15, 'HEALTHY LIVING': 16, 'HOME & LIVING': 17, 'IMPACT': 18, 'LATINO VOICES': 19, 'MEDIA': 20, 'MONEY': 21, 'PARENTING': 22, 'PARENTS': 23, 'POLITICS': 24, 'QUEER VOICES': 25, 'RELIGION': 26, 'SCIENCE': 27, 'SPORTS': 28, 'STYLE': 29, 'STYLE & BEAUTY': 30, 'TASTE': 31, 'TECH': 32, 'THE WORLDPOST': 33, 'TRAVEL': 34, 'U.S. NEWS': 35, 'WEDDINGS': 36, 'WEIRD NEWS': 37, 'WELLNESS': 38, 'WOMEN': 39, 'WORLD NEWS': 40, 'WORLDPOST': 41}
label_dict_inverted = {
    0: 'ARTS', 1: 'ARTS & CULTURE', 2: 'BLACK VOICES', 3: 'BUSINESS', 4: 'COLLEGE', 5: 'COMEDY', 6: 'CRIME',
    7: 'CULTURE & ARTS', 8: 'DIVORCE', 9: 'EDUCATION', 10: 'ENTERTAINMENT', 11: 'ENVIRONMENT', 12: 'FIFTY',
    13: 'FOOD & DRINK', 14: 'GOOD NEWS', 15: 'GREEN', 16: 'HEALTHY LIVING', 17: 'HOME & LIVING', 18: 'IMPACT',
    19: 'LATINO VOICES', 20: 'MEDIA', 21: 'MONEY', 22: 'PARENTING', 23: 'PARENTS', 24: 'POLITICS',
    25: 'QUEER VOICES', 26: 'RELIGION', 27: 'SCIENCE', 28: 'SPORTS', 29: 'STYLE', 30: 'STYLE & BEAUTY',
    31: 'TASTE', 32: 'TECH', 33: 'THE WORLDPOST', 34: 'TRAVEL', 35: 'U.S. NEWS', 36: 'WEDDINGS',
    37: 'WEIRD NEWS', 38: 'WELLNESS', 39: 'WOMEN', 40: 'WORLD NEWS', 41: 'WORLDPOST'
}
test = pd.DataFrame({
    "text" : data.authors+" "+data.headline+" "+data.link+" "+data.short_description,
    "label" : data.category
})
test_dataset = TextClassificationDataset(
    texts=test['text'].values.tolist(),
    labels=test['label'].values.tolist(),
    label_dict=label_dict,
    max_seq_length=MAX_SEQ_LENGTH,
    model_name=MODEL_NAME
)
# 加载训练好的模型
loaded_model = DistilBertForSequenceClassification(pretrained_model_name=MODEL_NAME,
                                                    num_classes=NUM_CLASSES)
loaded_model.load_state_dict(torch.load('results/trained_model_best.pth'))  # 替换成你想要加载的模型文件名

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model.to(device)

# 设置模型为评估模式
loaded_model.eval()

def print_classifications_test():
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            features = batch['features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = loaded_model(features, attention_mask)
            preds = torch.argmax(outputs, dim=1).item()
            predicted_classes = label_dict_inverted[preds]
            print(f"{batch_idx+1}.{predicted_classes}")

def cal_acc_test():
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = loaded_model(features, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)

            predicted_labels.extend(predicted.cpu().numpy())  # 预测的标签
            true_labels.extend(targets.cpu().numpy())  # 真实的标签

        # 直接比较预测标签和真实标签
        correct_samples = np.sum(np.array(predicted_labels) == np.array(true_labels))
        print(f"correct_samples: {correct_samples}, total_samples: {len(true_labels)}")

        acc = accuracy_score(true_labels, predicted_labels)  # 计算准确率
        precision = precision_score(true_labels, predicted_labels, zero_division=1, average='macro')  # 计算精确度
        f1 = f1_score(true_labels, predicted_labels, average='macro')  # 计算F1 Score
        print(f"Test Accuracy: {acc}, Test Precision: {precision}, Test F1 Score: {f1}")

if __name__ == "__main__":
    cal_acc_test()