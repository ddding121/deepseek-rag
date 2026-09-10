"""中文二元字组/英文词 BM25 + 语义检索，以 RRF 融合，可选 CrossEncoder 重排。"""
import math
import re
from collections import Counter, defaultdict
import numpy as np


def tokenize(text):
    tokens = re.findall(r'[a-z0-9]+(?:[._/-][a-z0-9]+)*', text.lower())
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        tokens.extend(run[i:i+2] for i in range(len(run)-1))
        if len(run)==1:
            tokens.append(run)
    return tokens


class BM25:
    def __init__(self, texts):
        docs=[Counter(tokenize(t)) for t in texts]
        self.lengths=np.array([sum(d.values()) for d in docs],dtype=float)
        self.avg=float(self.lengths.mean()) if docs else 1.0
        self.postings=defaultdict(list)
        for i,doc in enumerate(docs):
            for term,freq in doc.items():
                self.postings[term].append((i,freq))

    def scores(self, question):
        n=len(self.lengths)
        scores=np.zeros(n)
        for term in set(tokenize(question)):
            postings=self.postings.get(term,[])
            idf=math.log(1+(n-len(postings)+0.5)/(len(postings)+0.5))
            for i,freq in postings:
                norm=1.5*(1-0.75+0.75*self.lengths[i]/max(self.avg,1))
                scores[i]+=idf*freq*2.5/(freq+norm)
        return scores


def search(question,chunks,vectors,encoder,top_k=5,threshold=0.5,mode='hybrid',lexical=None,reranker=None,diagnostics=None):
    if not chunks:
        return [], []
    query=encoder.encode(['为这个句子生成表示以用于检索相关文章：'+question],
                         normalize_embeddings=True,convert_to_numpy=True)[0]
    semantic=np.asarray(vectors)@query
    if diagnostics is not None:
        diagnostics['max_semantic']=float(np.max(semantic))
    limit=min(len(chunks),max(20,top_k*4))
    dense=[int(i) for i in np.argsort(-semantic,kind='stable')[:limit] if semantic[i]>=threshold]
    lexical_scores=np.zeros(len(chunks))
    rankings=[dense]
    if mode=='hybrid':
        lexical_scores=(lexical or BM25([c.text for c in chunks])).scores(question)
        rankings.append([int(i) for i in np.argsort(-lexical_scores,kind='stable')[:limit] if lexical_scores[i]>0])
    elif mode!='dense':
        raise ValueError('未知检索模式')
    fused=defaultdict(float)
    for ranking in rankings:
        for rank,i in enumerate(ranking,1):
            fused[i]+=1/(60+rank)
    candidates=sorted(fused,key=lambda i:(-fused[i],i))[:limit]
    rerank_scores={}
    if reranker is not None and candidates:
        values=np.asarray(reranker.predict([(question,chunks[i].text) for i in candidates],
                          batch_size=8,show_progress_bar=False)).reshape(-1)
        if len(values)!=len(candidates) or not np.isfinite(values).all():
            raise ValueError('重排序模型返回了无效分数')
        rerank_scores=dict(zip(candidates,map(float,values)))
        candidates.sort(key=lambda i:-rerank_scores[i])
    selected=[]
    seen=set()
    for i in candidates:
        # 同一文件的完全相同片段只给模型一次，来源仍为选中片段的位置。
        key=(chunks[i].source,chunks[i].text.strip())
        if key in seen:
            continue
        seen.add(key)
        selected.append(i)
        if len(selected)>=top_k:
            break
    details=[dict(semantic=float(semantic[i]),bm25=float(lexical_scores[i]),
                  rrf=float(fused[i]),rerank=rerank_scores.get(i)) for i in selected]
    # 保持 answer 接口；第二个值始终为余弦相似度，不冒充混合分数或可信度。
    return [(chunks[i],float(semantic[i])) for i in selected],details


def similarity_reminder(maximum, threshold):
    if maximum is None or maximum >= threshold:
        return None
    return (f'本次最高语义相似度为 {maximum:.3f}，低于设定阈值 {threshold:.2f}。'
            '可以先核对相关原文，再适当降低左侧“语义候选最低相似度”，然后重新发送问题。'
            '降低阈值可能引入无关内容，程序不会自动修改设置。')
