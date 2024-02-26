with open('./content/nodes/nodes.tsv', 'r', encoding='utf-8') as file:
    c = file.readlines()


id = []
text = []
id1 = []
text1 = []
cnt = 0
for i in range(len(c)):
    cnt += 1
    try:
        data = c[i].split('\t')
        id.append(data[0].strip())
        text.append(data[1].strip())
        id1.append(data[0].strip())
        text1.append(data[1].strip())
    except:
        print(i)
        continue
text = text[1:]
id = id[1:]
text1 = text1[1:]
id1 = id1[1:]


save = []
for i in range(len(text)):
    if len(text[i]) == 0:
        save.append(id[i])


def remove_double_curly_braces(text):
    stack = []
    clean_text = ""

    for char in text:
        if char == "{":
            stack.append(char)
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        else:
            if not stack:
                clean_text += char
    return clean_text

def balance_curly_braces(text):
    opening_count = text.count("{")
    closing_count = text.count("}")

    if opening_count > closing_count:
        while opening_count > closing_count:
            index = text.find("{")
            if index != -1:
                text = text[:index] + text[index + 1:]
                opening_count -= 1
    elif closing_count > opening_count:
        while closing_count > opening_count:
            index = text.rfind("}")
            if index != -1:
                text = text[:index] + text[index + 1:]
                closing_count -= 1

    return text.strip()


from tqdm import tqdm
import re
cnt = 0
aaa = []
idd = []
for i in tqdm(range(len(text))):
    temp = text[i]
    text[i] = text[i].replace("'", '')

    # text[i] = re.sub(r'\{.*?\}', ' ', text[i])
    # text[i] = re.sub(r'\{\{.*?\}\}', ' ', text[i])
    # text[i] = re.sub(r'\{.*?\}', ' ', text[i])
    # if '}' in text[i] and text[i][-1] != '}' :
    #   while '}' in text[i]:
    #     u = text[i].find('}')
    #     text[i] = str(text[i][u+2:].strip())

    # if len(text[i]) == 0 :
    #   print(i)
    #   print(temp)
    #   text[i] = temp
    text[i] = balance_curly_braces(text[i])
    text[i] = remove_double_curly_braces(text[i])
    text[i] = re.sub(r"<!--.*?-->", ' ', text[i])
    if len(text[i]) == 0:
        cnt += 1
        text[i] = temp
    # text[i] = re.sub(r']*>', ' ', text[i])
    # text[i] = aaa
    text[i] = re.sub(r'\[.*?\]', ' ', text[i])
    text[i] = re.sub(r"'''", ' ', text[i])
    text[i] = re.sub(r"''", ' ', text[i])
    # text[i] = re.sub(r"", ' ', text[i])
    text[i] = re.sub(r"\([,./;\']+\)", ' ', text[i])
    text[i] = re.sub(r"\{\{\s*cite\s*.*?\}\}", ' ', text[i])
    text[i] = re.sub(r'\{ class="[^"]*"[\s\S]*?1 \+ \}', ' ', text[i])

    text[i] = text[i].replace('`', ' ').replace('(', ' ').replace(')', ' ').replace('*', ' ').replace(';', ' ').replace(
        '’', ' ')
    text[i] = text[i].replace('‘', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ')
    text[i] = text[i].replace('‘', ' ').replace(',', ' ').replace('.', ' ').replace('#', ' ').replace('"', ' ')
    text[i] = re.sub(r'\s+', ' ', text[i])

    # tokens = word_tokenize(text[i])
    # filtered_tokens = [word for word in tokens if word.casefold() not in stop_words]
    # text[i] = ' '.join(filtered_tokens)

    # text[i] = aaa''


data = dict(zip(id, text))
import pandas as pd

train = pd.read_csv('./content/train.csv')
sentence1 = []
sentence2 = []
for i in range(len(train)):
    id1 = str(train['id1'][i])
    id2 = str(train['id2'][i])
    sentence1.append(data[id1])
    sentence2.append(data[id2])
train['sentence1'] = sentence1
train['sentence2'] = sentence2
train.to_csv('./content/my_train.csv', index=False)

test = pd.read_csv('./content/test.csv')
sentence1 = []
sentence2 = []
for i in range(len(test)):
    id1 = str(test['id1'][i])
    id2 = str(test['id2'][i])
    sentence1.append(data[id1])
    sentence2.append(data[id2])
test['sentence1'] = sentence1
test['sentence2'] = sentence2
test.to_csv('./content/my_test.csv', index=False)
with open('hi.txt', 'w', encoding='utf-8') as file:
    for item in text:
        file.write("%s\n" % item)
