import tempfile
import unittest
from pathlib import Path
import numpy as np
from ingestion import ingest, pipeline_key, plan_uploads
from rag import Chunk
from storage import Store

class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'knowledge.db'
        self.store=Store(self.path)
        self.kb=self.store.create_kb('课程','test-model')
        self.calls=[]
    def tearDown(self):
        self.temp.cleanup()
    def add(self,uploads,replace=False,parser=None):
        def parse(name,data,report):
            self.calls.append(name)
            return [Chunk(name,'段落 1',data.decode())],[]
        return ingest(self.store,self.kb,uploads,pipeline_key(True,'auto'),replace,parser or parse,
                      lambda chunks,report:np.array([[1.,0.]]*len(chunks)),lambda v,m:None)
    def record(self,name='a.docx',text='new',vectors=None):
        return dict(name=name,data=text.encode(),pipeline='p',chunks=[Chunk(name,'1',text)],
                    vectors=np.array([[1.,0.]]) if vectors is None else vectors,notes=[])
    def test_restart(self):
        self.add([('a.docx',b'original')])
        reopened=Store(self.path)
        kb,docs=reopened.snapshot(self.kb)
        chunks,vectors=reopened.load_index(self.kb,kb['revision'])
        self.assertEqual(chunks[0].text,'original')
        self.assertEqual(reopened.document(self.kb,docs[0]['id'])['original'],b'original')
        np.testing.assert_array_equal(vectors,[[1,0]])
    def test_incremental(self):
        self.add([('a.docx',b'a')]);self.calls.clear()
        result=self.add([('a.docx',b'a'),('b.docx',b'b')])
        self.assertEqual(self.calls,['b.docx'])
        self.assertEqual(result,(1,['a.docx']))
    def test_replace(self):
        self.add([('a.docx',b'old')])
        with self.assertRaises(ValueError):self.add([('a.docx',b'new')])
        self.add([('a.docx',b'new')],True)
        chunks,_=self.store.load_index(self.kb,2)
        self.assertEqual([c.text for c in chunks],['new'])
    def test_parser_failure(self):
        self.add([('a.docx',b'old')])
        def parse(name,data,report):
            if name=='bad.docx':raise RuntimeError('bad')
            return [Chunk(name,'1','new')],[]
        with self.assertRaises(RuntimeError):
            self.add([('a.docx',b'new'),('bad.docx',b'bad')],True,parse)
        self.assertEqual(self.store.load_index(self.kb,1)[0][0].text,'old')
    def test_delete(self):
        self.add([('a.docx',b'a')])
        kb,docs=self.store.snapshot(self.kb)
        self.store.delete_document(self.kb,docs[0]['id'],kb['revision'])
        with self.assertRaises(ValueError):self.store.load_index(self.kb,1)
        self.assertEqual(self.store.load_index(self.kb,2)[0],[])
    def test_separate_kbs_and_backup(self):
        self.add([('a.docx',b'a')])
        other=self.store.create_kb('第二个','test-model')
        self.assertEqual(self.store.load_index(other,0)[0],[])
        backup=Path(self.temp.name)/'backup.db';self.store.backup(backup)
        restored=Store(backup)
        self.assertEqual(len(restored.list_kbs()),2)
        self.assertEqual(restored.load_index(self.kb,1)[0][0].text,'a')
    def test_invalid_vector_rolls_back_entire_batch(self):
        self.add([('a.docx',b'old')])
        with self.assertRaises(ValueError):
            self.store.save_batch(self.kb,1,[self.record(),self.record('b.docx',vectors=np.array([[np.nan,0]]))],True)
        self.assertEqual(self.store.load_index(self.kb,1)[0][0].text,'old')
    def test_concurrent_write_guard(self):
        self.add([('a.docx',b'a')]);self.add([('b.docx',b'b')])
        with self.assertRaises(ValueError):self.store.save_batch(self.kb,1,[self.record()],True)
        self.assertEqual(self.store.load_index(self.kb,2)[0][0].text,'a')
    def test_changed_ocr_config(self):
        self.add([('a.docx',b'a')]);_,docs=self.store.snapshot(self.kb)
        with self.assertRaises(ValueError):
            plan_uploads(docs,[('a.docx',b'a')],pipeline_key(True,'all'),False)
    def test_quota_rollback(self):
        records=[self.record(f'{i}.docx') for i in range(11)]
        with self.assertRaises(ValueError):self.store.save_batch(self.kb,0,records)
        self.assertEqual(self.store.snapshot(self.kb)[1],[])

if __name__=='__main__':unittest.main()
