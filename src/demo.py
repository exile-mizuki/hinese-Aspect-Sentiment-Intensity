"""从训练集拟合已选配置，为任意评论中的指定方面打分。"""
import argparse
from main import train, cols, local
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
import numpy as np

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('sentence'); parser.add_argument('aspect',nargs='+')
    args=parser.parse_args(); t,_=train()
    query=[[args.sentence]*len(args.aspect),args.aspect,[local(args.sentence,a) for a in args.aspect]]
    train_blocks=[]; query_blocks=[]
    for texts,new in zip(cols(t),query):
        v=TfidfVectorizer(analyzer='char',ngram_range=(1,3),sublinear_tf=True)
        train_blocks.append(v.fit_transform(texts));query_blocks.append(v.transform(new))
    m=Ridge(alpha=.1,solver='lsqr',tol=1e-5).fit(hstack(train_blocks).tocsr(),t[['V','A']])
    for a,(v,h) in zip(args.aspect,np.clip(m.predict(hstack(query_blocks).tocsr()),1,9)):
        print(f'{a}: V={v:.2f}  A={h:.2f}')
