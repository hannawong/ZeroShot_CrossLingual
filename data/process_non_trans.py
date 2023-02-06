import pandas as pd
from transformers import MBartConfig, MBartForConditionalGeneration,AutoTokenizer
MODEL_CLASSES = {
    'mbart': (MBartConfig, MBartForConditionalGeneration,AutoTokenizer)
}

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25") ##None

for splits in ["train","dev","test"]:
    for lang in ["ZH","DE","ES"]:
        df = pd.read_csv(splits+"_"+lang+".tsv",sep="\t")
        slot_labels = [item.split(" ") for item in list(df["slot_labels"])]
        utterance = [tokenizer.tokenize(item) for item in list(df["utterance"])]
        for i in range(len(slot_labels)):
            print(len(slot_labels[i]),len(utterance[i]))
            if len(slot_labels[i]) != len(utterance[i]):
                print(slot_labels[i],utterance[i])
                exit()
        