import spacy
from torchtext.data import Field, BucketIterator
from dataloader import Multi30k, WMT14

def preprocess():
    spacy_en = spacy.load('en_core_web_sm') # 영어 토큰화(tokenization)
    spacy_de = spacy.load('de_core_news_sm') # 독일어 토큰화(tokenization)

    # 간단히 토큰화(tokenization) 기능 써보기
    tokenized = spacy_en.tokenizer("I am a graduate student.")

    for i, token in enumerate(tokenized):
        print(f"인덱스 {i}: {token.text}")


    # 독일어(Deutsch) 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)

    # def tokenize_de(text):
    #    return [token.text for token in spacy_de.tokenizer(text)]

    # # 영어(English) 문장을 토큰화 하는 함수
    # def tokenize_en(text):
    #    return [token.text for token in spacy_en.tokenizer(text)]
    
    def tokenize_space(text):
        return text.split(' ')

    SRC = Field(tokenize=tokenize_space, init_token="", eos_token="", lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_space, init_token="", eos_token="", lower=True, batch_first=True)

    # train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))
    train_dataset, valid_dataset, test_dataset = WMT14.splits(exts=(".de", ".en"), fields=(SRC, TRG))

    print(f"학습 데이터셋(training dataset) 크기: {len(train_dataset.examples)}개")
    print(f"평가 데이터셋(validation dataset) 크기: {len(valid_dataset.examples)}개")
    print(f"테스트 데이터셋(testing dataset) 크기: {len(test_dataset.examples)}개")

    # 학습 데이터 중 하나를 선택해 출력
    print(vars(train_dataset.examples[30])['src'])
    print(vars(train_dataset.examples[30])['trg'])

    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    print(f"len(SRC): {len(SRC.vocab)}")
    print(f"len(TRG): {len(TRG.vocab)}")

    print(TRG.vocab.stoi["abcabc"]) # 없는 단어: 0
    print(TRG.vocab.stoi[TRG.pad_token]) # 패딩(padding): 1
    print(TRG.vocab.stoi[""]) # : 2
    print(TRG.vocab.stoi[""]) # : 3
    print(TRG.vocab.stoi["hello"])
    print(TRG.vocab.stoi["world"])

    return SRC, TRG, train_dataset, valid_dataset, test_dataset