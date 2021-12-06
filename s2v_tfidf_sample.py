# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
import s2v_tfidf
from scipy import spatial


def main():

    corpus=['今日は暑かった．辛いものも食べた．とても汗をかいたけど，楽しかった．','なんて日だ！，これからあんなとこ行くんじゃないわ！二度と行かない！','次元数を操作する必要がある例として配列の転置の例を紹介します．']
    model_dir = './tohoku_entity_vector/entity_vector/entity_vector.model.bin'
    model = KeyedVectors.load_word2vec_format(model_dir, binary=True)

    s2v = s2v_tfidf.s2vtfidf(model)

    wakachi_corpus = s2v.make_wakachi_corpus(corpus)
    vec,X_tfidf = s2v.TFIDF(wakachi_corpus)
    corpus_vec = s2v.tfidf_sentence_vector(vec,X_tfidf)
    print('0 and 1',1 - spatial.distance.cosine(corpus_vec[0], corpus_vec[1]))
    print('0 and 2',1 - spatial.distance.cosine(corpus_vec[0], corpus_vec[2]))
    print('1 and 2',1 - spatial.distance.cosine(corpus_vec[1], corpus_vec[2]))


if __name__ == '__main__':
    main()
