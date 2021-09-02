import pymongo

client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["re_py150"]
train_col = db["train"]
train_col.remove()

cnt = 1
limit = 100000000

code=open("/home/v-weixyan/Desktop/CodeXGLUE/Code-Code/CodeCompletion-token/dataset/py150/token_completion/train.txt","r",encoding="utf-8")
code_list=code.readlines()
for item in enumerate(code_list):
    #print(item)
    #print(type(item))
    mydict = {"id":item[0], "code":item[1]}
    if cnt == limit:
        break
    cnt += 1
    train_col.insert_one(mydict)
