# sen2vec
- It calculates Corpus tfidf values and, based on the weights. The module obtains the distributed representation of each sentence by adding the distributed representations of words together.
- You can obtain a distributed representation of each sentence with this.

## Word Embedding
- I used Tohoku University's Japanese Wikipedia entity vector to give the distributed representation of words.
  - Reference: http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
  - 20161101.tar.bz2 (November 1, 2016 version, 1.3 GB, 2.6 GB after unzipping) file

## Tool
- Morphological analyzer: Janome

## Note
- If a word in the sentence is not in the dictionary of distributed representations, the distributed representation of the sentence is computed without considering the word.

## How to use
- Please refer to s2v_tfidf_sample.py


