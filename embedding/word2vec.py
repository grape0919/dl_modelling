#import nltk
# nltk.download('punkt')

from os import WIFCONTINUED
from lxml import etree
from nltk import tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from gensim.models import Word2Vec, KeyedVectors
import urllib.request
import pandas as pd
from konlpy.tag import Okt
#urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")
#
#print("End downloading train data ")
def W2V_for_English():
    # '''############# 전처리 ############'''
    # targetXML = open('ted_en-20160408.xml', 'r', encoding="UTF8")

    # # print("!@#!@#origin data : ", targetXML.read())
    # target_text = etree.parse(targetXML)

    # print("!@#!@# target_text : ", target_text)

    # parse_text = '\n'.join(target_text.xpath('//content/text()'))

    # content_text = re.sub(r'\([^)]*\)', '', parse_text)

    # sent_text = sent_tokenize(content_text)

    # normalize_text = []
    # for string in sent_text:
    #     toekns = re.sub(r"[^a-z0-9]+", " ", string.lower()) #영어 숫자 아닌것은 모두 제거 후 소문자 치환
    #     normalize_text.append(toekns)

    # result = [word_tokenize(sentence) for sentence in normalize_text]

    # print(f'총 샘플의 개수 : {len(result)}')

    '''################################'''


    '''############# Embedding ############'''
    # for line in result[:3]:
    #     print(line)


    # model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)

    '''
    size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
    window = 컨텍스트 윈도우 크기
    min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
    workers = 학습을 위한 프로세스 수
    sg = 0은 CBOW, 1은 Skip-gram.
    '''
    # 비슷한 단어 출력
    model = KeyedVectors.load_word2vec_format("model/eng_w2v") # 모델 로드
    model_result = model.most_similar("man")
    print(model_result)
    '''################################'''

    '''############ Save Model #############'''
    # model.wv.save_word2vec_format('model/eng_w2v') # 모델 저장
    # loaded_model = KeyedVectors.load_word2vec_format("model/eng_w2v") # 모델 로드

def W2V_for_Korean():
    #네이버 영화 리뷰
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
    train_data = pd.read_table('ratings.txt')

    # remove null data
    train_data = train_data.dropna(how = 'any')

    # 한글자 제거
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

    # 불용어 설정
    # stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

    okt = Okt()
    tokenized_data = []
    for sentence in train_data['document']:
        temp_X = okt.morphs(sentence, stem=True)
        temp_Y = [word for word in temp_X]# if not word in stopwords]
        tokenized_data.append(temp_Y)
    
    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=5, workers=4, sg=0)
    model.wv.save_word2vec_format("model/kor_w2v")
    
    model = KeyedVectors.load_word2vec_format('model/kor_w2v')
    print(model.most_similar("최민식"))
    print(model.most_similar("장르"))


if __name__ == '__main__':
    #영어 Word2Vec
    # W2V_for_English()

    #한글 Word2Vec
    W2V_for_Korean()