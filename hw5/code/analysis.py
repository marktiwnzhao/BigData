import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

data_path = "./data"
data = pd.read_json(data_path+"/News_Category.json", lines=True)

print(f"Number of Samples: {data.shape[0]}")
print(f"Number of Features: {data.shape[1]}")

# 统计每个类别的样本数量
category_counts = data['category'].value_counts()
category_counts.plot(kind='barh', figsize=(15,10))
# 在每个条形上显示样本数量
for i, count in enumerate(category_counts):
    plt.text(count, i, str(count), ha='left', va='center', fontweight='bold')
plt.xlabel("Total")
plt.ylabel("Category")
plt.title("Figure 1 - Category Distribution on News",
          fontweight="bold", size=12)
plt.show()

# 绘制饼状图
plt.figure(figsize=(15, 15))
category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # 保证饼状图比例尺寸相等，使其为圆形
plt.title("Figure 2 - Category Distribution on News (Pie Chart)",
          fontweight="bold", size=12)
plt.ylabel('')  # 去除y轴标签
plt.show()

# 将所有描述合并成一个长文本
text = ' '.join(data['short_description'].dropna().astype(str).tolist())
# 生成词云
wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(text)
# 显示词云图
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 去除坐标轴
plt.title("Figure 3 - Word Cloud for Short Descriptions",
          fontweight="bold", size=12)
plt.show()