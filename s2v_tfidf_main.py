# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
import numpy as np
from scipy import spatial

# corpus=["This is a dog.","Hello World!","This is a cat."]
# corpus=["This is a dog.","Hello World!","This is a cat."]
# corpus=['世の中には面白い考え方をする人間がいるもので、中には「公衆の面前であえてメンツを潰すように周囲に聞こえよがしに叱責すると言われた人間は恐縮して以後言うことを聞くようになる」と本気で信じている人間がいるらしい。ひとつ言えることは、それをやって平気な当人が異常人格だということかと','誰もが発信者となった現代社会において、私たちが日々読むことのできる情報の量は爆発的に増加しています。一つだけ確かなことは、それらすべての情報を読むことは決してできない、私たちは読むべき情報とそうでない情報を日々選び取らなければならない、ということです。そうした環境のなかでも、より多くの人たちが、より良質な情報に出会ってほしい、さらにそれによって良質な情報の作り手が損をしない社会を作りたい。その思いで私たちはをつくり、様々な進化を重ねてきました。','世界中の膨大な情報を日夜解析し続けるアルゴリズムと、スマートデバイスに最適化された快適なインターフェースを通じて、smartnewsは世界中から集めた良質な情報を、一人でも多くの人々に届けていきたいと考えています。','世の中には面白い考え方をする人間がいるもので、中には「公衆の面前であえてメンツを潰すように周囲に聞こえよがしに叱責すると言われた人間は恐縮して以後言うことを聞くようになる」と本気で信じている人間がいるらしい。ひとつ言えることは、それをやって平気な当人が異常人格だということかと','今日の花火大会は楽しかった']

corpus=['今日は暑かった．辛いものも食べた．とても汗をかいたけど，楽しかった．','なんて日だ！，これからあんなとこ行くんじゃないわ！二度と行かない！','次元数を操作する必要がある例として配列の転置の例を紹介します．']

model_dir = './tohoku_entity_vector/entity_vector/entity_vector.model.bin'
model = KeyedVectors.load_word2vec_format(model_dir, binary=True)

'''
入力:文字列
出力:分かち書きされた文字列
'''
def Wakachi(sentence):#文字列を分かち書きする
    words=[]
    t = Tokenizer()
    for token in t.tokenize(sentence):#janomeで分かち書きをする．
        words.append(token.surface)
    Wakachi_sentence = ' '.join(words)
    return Wakachi_sentence

'''
入力:リスト型の複数文書(分かち書き済み)が格納された配列
出力:TfidfVectorizerのインスタンス，与えたコーパスのTFIDF値が入ったインスタンス
'''
def TFIDF(corpus):
    vec = TfidfVectorizer(min_df=1)
    X_tfidf = vec.fit_transform(corpus)
    return vec,X_tfidf#vecには単語情報が，X_tfidfには，TFIDF値が入っている．

'''
入力:リスト型の複数文書が格納された配列
出力:配列内の文字列がそれぞれ分かち書きされた(半角スペース区切り)テキストが入った配列．
'''
def make_wakachi_corpus(corpus):
    wakachi_corpus = []
    for text in corpus:
        wakachi_text = Wakachi(text)
        wakachi_corpus.append(wakachi_text)
    return wakachi_corpus

'''
入力:TFIDFの単語の情報が入ったインスタンス(vec)，tfidf値が入ったインスタンス(X_tfidf)，分散表現の学習済みモデル(model)
出力:
'''
def tfidf_sentence_vector(vec,X_tfidf,model):
    corpus_vec=[]
    num_features=200
    features = vec.get_feature_names()
    print(features)
    print(len(features))
    feature_vec = np.zeros((num_features,), dtype="float32") # 特徴ベクトルの入れ物を初期化
    # print('len(X_tfidf.toarray())>>',len(X_tfidf.toarray()))
    print(X_tfidf.toarray())
    # exit()
    for i in range(len(X_tfidf.toarray())):
        for j,feature in enumerate(features):#TFIDFインスタンス内の単語の数だけ，足し合わせる
            # print('X_tfidf.toarray()['+str(i)+']['+str(j)+']>>',X_tfidf.toarray()[i][j],'feature>>',feature)
            try:#辞書にないものは無視して，文章の分散表現を計算する．
                feature_vec = np.add(feature_vec,model[feature]*X_tfidf.toarray()[i][j])
            except KeyError:
                # print(feature,'is not found in dictionary')
                feature_vec = feature_vec#何もしない
        # print(feature_vec)
        corpus_vec.append(feature_vec)

    # print('len(feature_vec))>>',len(feature_vec))
    # print('corpus_vec.shape>>',corpus_vec.shape)
    print('corpus_vec.shape>>',np.array(corpus_vec).shape)
    return corpus_vec

def main():
    wakachi_corpus = make_wakachi_corpus(corpus)
    # print(wakachi_corpus)
    # exit()
    vec,X_tfidf = TFIDF(wakachi_corpus)
    corpus_vec = tfidf_sentence_vector(vec,X_tfidf,model)
    print('0 and 1',1 - spatial.distance.cosine(corpus_vec[0], corpus_vec[1]))
    print('0 and 2',1 - spatial.distance.cosine(corpus_vec[0], corpus_vec[2]))
    print('1 and 2',1 - spatial.distance.cosine(corpus_vec[1], corpus_vec[2]))
    # print('0 and 3',1 - spatial.distance.cosine(corpus_vec[0], corpus_vec[3]))
    # print('0 and 4',1 - spatial.distance.cosine(corpus_vec[0], corpus_vec[4]))


if __name__ == '__main__':
    main()
