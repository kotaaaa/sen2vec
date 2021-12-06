# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
import sys
model_dir = './tohoku_entity_vector/entity_vector/entity_vector.model.bin'
model = KeyedVectors.load_word2vec_format(model_dir, binary=True)

# 入力された単語から近い単語をn個表示する
def similarWords(posi, nega=[], n=10):
    cnt = 1 # 表示した単語の個数カウント用
    # 学習済みモデルからcos距離が最も近い単語n個(topn個)を表示する
    result = model.most_similar(positive = posi, negative = nega, topn = n)
    for r in result:
        print('Top10 Similar',cnt,' ', r[0],' ', r[1])
        cnt += 1

def main():
    word = sys.argv[1]
    similarWords([word])
    print('model[word]\n',model[word],len(model[word]))

if __name__ == '__main__':
    main()
