# py150数据集

具体下载方式及处理方式上级说明中已说明.

data_input是py150数据集中截取和过滤完做了inference的train data和test data的input.

## py150_train

input.txt是py150数据集的train data的每条数据按行截取一半之后并过滤了过短数据之后的结果，从95000条data过滤成了88154条data.

train_input.json是上个文件中截取过滤之后的data及他们原始的id.

train_gt.txt是每条data截取完之后的next line，也就是每条数据的ground truth.

train_pred.txt是每条data在微调完的CodeGPT中的inference生成的下一行code.

train_recitation.json是train data做inference之后的结果在train data语料中的recitation行为，即best_simi_code_dis=0的情况.

train_data_recitiation_input.txt是在train_recitation.json中筛选了出现次数小于10，且能得到该recitiation code结果的input条数大于1的结果.

train_data_recitiation_act.json记录了recitiation code及id和相对应的input及id.

train_input_activations.json在train_data_recitiation_act.json基础上增加记录了每个input的12*768个neuron的activations.

## py150_test

同理，不过是test data.


## py150
下载之后的处理完的原始数据.
