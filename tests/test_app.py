import tempfile
import unittest
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
from storage import Store
from rag import Chunk

class AppTests(unittest.TestCase):
    def test_create_reopen_and_delete_saved_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'app.py'
            path.write_text((Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8'),encoding='utf-8')
            at=AppTest.from_file(str(path)).run(timeout=20)
            self.assertEqual(len(at.exception),0)
            next(x for x in at.text_input if x.label=='知识库名称').set_value('测试资料')
            next(x for x in at.button if x.label=='创建').click()
            at.run(timeout=20)
            self.assertEqual(len(at.exception),0)
            store=Store(Path(directory)/'data'/'knowledge.db')
            kb=store.list_kbs()[0]
            store.save_batch(kb['id'],0,[dict(name='test.docx',data=b'test',pipeline='p',
                              chunks=[Chunk('test.docx','段落 1','saved text')],vectors=np.array([[1.,0.]]),notes=[])])
            reopened=AppTest.from_file(str(path)).run(timeout=20)
            self.assertEqual(len(reopened.exception),0)
            self.assertTrue(any(x.value=='saved text' for x in reopened.text))
            next(x for x in reopened.checkbox if x.label.startswith('确认从当前知识库删除')).check()
            reopened.run()
            next(x for x in reopened.button if x.label=='删除选中文档').click()
            reopened.run()
            self.assertEqual(len(reopened.exception),0)
            self.assertEqual(store.snapshot(kb['id'])[1],[])

if __name__=='__main__':unittest.main()
