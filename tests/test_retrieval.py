import unittest
import numpy as np
from rag import Chunk
from retrieval import BM25, tokenize, search
from chunking import split_semantic

class Encoder:
    def encode(self,*args,**kwargs):return np.array([[1.,0.]])
class Reranker:
    def predict(self,pairs,**kwargs):return np.array([1. if 'ZX-900' in text else 0. for q,text in pairs])

class RetrievalTests(unittest.TestCase):
    def test_identifiers_and_chinese(self):
        tokens=tokenize('ZX-900 第十二条 请假制度')
        self.assertIn('zx-900',tokens);self.assertIn('请假',tokens)
    def test_keyword_recovers_low_semantic_identifier(self):
        chunks=[Chunk('a','1','普通设备说明'),Chunk('b','2','ZX-900 故障时断电重启')]
        vec=np.array([[1.,0.],[0.,1.]])
        dense,_=search('ZX-900',chunks,vec,Encoder(),threshold=.5,mode='dense')
        hybrid,details=search('ZX-900',chunks,vec,Encoder(),threshold=.5,mode='hybrid')
        self.assertNotIn('b',[c.source for c,s in dense])
        self.assertIn('b',[c.source for c,s in hybrid])
        self.assertTrue(any(d['bm25']>0 for d in details))
    def test_rerank_order(self):
        chunks=[Chunk('a','1','普通设备说明'),Chunk('b','2','ZX-900 故障时断电重启')]
        hits,details=search('ZX-900',chunks,np.eye(2),Encoder(),reranker=Reranker())
        self.assertEqual(hits[0][0].source,'b')
        self.assertEqual(details[0]['rerank'],1)
    def test_no_evidence(self):
        hits,_=search('火星旅行', [Chunk('a','1','员工请假')],np.array([[0.,1.]]),Encoder())
        self.assertEqual(hits,[])
    def test_dedup_and_topk(self):
        chunks=[Chunk('a',str(i),'相同内容') for i in range(3)]
        hits,_=search('内容',chunks,np.array([[1.,0.]]*3),Encoder())
        self.assertEqual(len(hits),1)
    def test_heading_and_paragraph_pack(self):
        text='# 请假制度\n试用期员工请假须直属经理审批。\n请提前一天申请。'
        parts=list(split_semantic(text))
        self.assertEqual(len(parts),1)
        self.assertIn('直属经理',parts[0]);self.assertIn('提前一天',parts[0])
    def test_length_bound_and_long_text_coverage(self):
        text='# 第一章\n'+''.join(chr(0x4e00+i) for i in range(900))
        parts=list(split_semantic(text))
        self.assertTrue(all(len(p)<=350 for p in parts))
        combined=''.join(parts)
        self.assertTrue(all(char in combined for char in text.split('\n')[1]))
    def test_different_headings_do_not_mix(self):
        parts=list(split_semantic('# 请假\n提前申请。\n# 报销\n保留发票。'))
        self.assertEqual(len(parts),2)
        self.assertNotIn('发票',parts[0])

if __name__=='__main__':unittest.main()
