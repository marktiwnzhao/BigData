import pandas as pd
import torch
from torch.utils.data import DataLoader
# SKlearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from some_classes import TextClassificationDataset, DistilBertForSequenceClassification
import logging
import os

data_path = "./data"
data = pd.read_json(data_path+"/News_Category.json", lines=True)

# 定义参数
MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 5
BATCH_SIZE = 80
MAX_SEQ_LENGTH = 256
LEARN_RATE = 5e-5
SEED = 42
LOG_DIR = 'results'

label_dict = {'ARTS': 0, 'ARTS & CULTURE': 1, 'BLACK VOICES': 2, 'BUSINESS': 3, 'COLLEGE': 4, 'COMEDY': 5, 'CRIME': 6, 'CULTURE & ARTS': 7, 'DIVORCE': 8, 'EDUCATION': 9, 'ENTERTAINMENT': 10, 'ENVIRONMENT': 11, 'FIFTY': 12, 'FOOD & DRINK': 13, 'GOOD NEWS': 14, 'GREEN': 15, 'HEALTHY LIVING': 16, 'HOME & LIVING': 17, 'IMPACT': 18, 'LATINO VOICES': 19, 'MEDIA': 20, 'MONEY': 21, 'PARENTING': 22, 'PARENTS': 23, 'POLITICS': 24, 'QUEER VOICES': 25, 'RELIGION': 26, 'SCIENCE': 27, 'SPORTS': 28, 'STYLE': 29, 'STYLE & BEAUTY': 30, 'TASTE': 31, 'TECH': 32, 'THE WORLDPOST': 33, 'TRAVEL': 34, 'U.S. NEWS': 35, 'WEDDINGS': 36, 'WEIRD NEWS': 37, 'WELLNESS': 38, 'WOMEN': 39, 'WORLD NEWS': 40, 'WORLDPOST': 41}
log_file = os.path.join(LOG_DIR, 'training_logs.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

text = pd.DataFrame({
    "text" : data.authors+" "+data.headline+" "+data.link+" "+data.short_description,
    "label" : data.category
})

# 划分数据集
train, val = train_test_split(text, test_size=0.20, random_state=SEED)

train_dataset = TextClassificationDataset(
    texts=train['text'].values.tolist(),
    labels=train['label'].values.tolist(),
    label_dict=label_dict,
    max_seq_length=MAX_SEQ_LENGTH,
    model_name=MODEL_NAME
)
valid_dataset = TextClassificationDataset(
    texts=val['text'].values.tolist(),
    labels=val['label'].values.tolist(),
    label_dict=label_dict,
    max_seq_length=MAX_SEQ_LENGTH,
    model_name=MODEL_NAME
)

NUM_CLASSES = len(label_dict)
train_val_loaders = {
    "train": DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True),
    "valid": DataLoader(dataset=valid_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)
}

model = DistilBertForSequenceClassification(pretrained_model_name=MODEL_NAME,
                                            num_classes=NUM_CLASSES)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(NUM_EPOCHS):
    # Training loop
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_val_loaders['train']):
        optimizer.zero_grad()

        features = batch['features'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        outputs = model(features, attention_mask=attention_mask)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                  f"Batch [{batch_idx + 1}/{len(train_val_loaders['train'])}] "
                  f"Loss: {loss.item()}")
            logging.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                  f"Batch [{batch_idx + 1}/{len(train_val_loaders['train'])}] "
                  f"Loss: {loss.item()}")

    average_loss = total_loss / len(train_val_loaders['train'])
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
          f"Average training loss: {average_loss}")
    logging.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
          f"Average training loss: {average_loss}")

    # Validation loop
    model.eval()
    val_loss = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for val_batch in train_val_loaders['valid']:
            features = val_batch['features'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            targets = val_batch['targets'].to(device)

            outputs = model(features, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            predicted_labels.extend(predicted.cpu().numpy())  # 预测的标签
            true_labels.extend(targets.cpu().numpy())  # 真实的标签

    average_val_loss = val_loss / len(train_val_loaders['valid'])
    acc = accuracy_score(true_labels, predicted_labels)  # 计算准确率
    precision = precision_score(true_labels, predicted_labels, average='macro')  # 计算精确度
    f1 = f1_score(true_labels, predicted_labels, average='macro')  # 计算F1 Score
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
          f"Test Accuracy: {acc}, Test Precision: {precision}, Test F1 Score: {f1}")
    logging.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
          f"Test Accuracy: {acc}, Test Precision: {precision}, Test F1 Score: {f1}")

    # 根据验证损失调整学习率
    scheduler.step(average_val_loss)
    # 保存训练好的模型
    torch.save(model.state_dict(), f'results/trained_model_epoch_{epoch + 1}.pth')