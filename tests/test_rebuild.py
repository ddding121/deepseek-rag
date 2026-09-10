import tempfile
import unittest
from pathlib import Path
import numpy as np
from storage import Store
from rag import Chunk
from ingestion import ingest,pipeline_key

class RebuildTests(unittest.TestCase):
    def test_old_db_rebuild_and_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'knowledge.db');kb=store.create_kb('旧库','model')
            store.save_batch(kb,0,[dict(name='a.docx',data=b'original',pipeline='ocr-v2',
                                      chunks=[Chunk('a.docx','1','old chunk')],vectors=np.array([[1.,0.]]),notes=[])])
            parse=lambda name,data,report:([Chunk(name,'1','new chunk')],[])
            encode=lambda chunks,report:np.array([[0.,1.]])
            ingest(store,kb,[('a.docx',b'original')],pipeline_key(True,'auto'),True,parse,encode,
                   lambda v,m:None,expected_revision=1,force=True)
            self.assertEqual(store.load_index(kb,2)[0][0].text,'new chunk')
            with self.assertRaises(ValueError):
                ingest(store,kb,[('a.docx',b'original')],pipeline_key(True,'auto'),True,parse,encode,
                       lambda v,m:None,expected_revision=1,force=True)
            ingest(store,kb,[('a.docx',b'original')],pipeline_key(True,'auto'),True,parse,encode,
                   lambda v,m:None,expected_revision=2,force=True)
            self.assertEqual(store.snapshot(kb)[0]['revision'],3)
