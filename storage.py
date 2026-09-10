"""SQLite 持久知识库。原文件、片段和向量在同一事务写入，失败保留旧版本。"""
import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from rag import Chunk

MAX_BYTES = 50 * 1024 * 1024


class Store:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError('数据库版本不兼容，请使用对应程序版本')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                    model TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                    name TEXT NOT NULL, name_key TEXT NOT NULL, digest TEXT NOT NULL,
                    pipeline TEXT NOT NULL, original BLOB NOT NULL, size INTEGER NOT NULL,
                    chunks TEXT NOT NULL, vectors BLOB NOT NULL, rows INTEGER NOT NULL,
                    dimension INTEGER NOT NULL, notes TEXT NOT NULL,
                    updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(kb_id, name_key));
                PRAGMA user_version=1;
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create_kb(self, name, model):
        name, model = name.strip(), model.strip()
        if not name or len(name) > 80 or not model:
            raise ValueError('知识库名称需为 1–80 字，模型名称不能为空')
        kb_id = uuid.uuid4().hex
        try:
            with self.connect() as db:
                db.execute('INSERT INTO knowledge_bases(id,name,model) VALUES (?,?,?)', (kb_id, name, model))
        except sqlite3.IntegrityError as exc:
            raise ValueError('已有同名知识库，请换一个名称') from exc
        return kb_id

    def list_kbs(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM knowledge_bases ORDER BY created,id')]

    def snapshot(self, kb_id):
        with self.connect() as db:
            db.execute('BEGIN')
            kb = db.execute('SELECT * FROM knowledge_bases WHERE id=?', (kb_id,)).fetchone()
            if kb is None:
                raise ValueError('知识库不存在，请刷新页面')
            docs = [dict(r) for r in db.execute(
                'SELECT id,name,name_key,digest,pipeline,size,rows,dimension,notes,updated FROM documents WHERE kb_id=? ORDER BY name', (kb_id,))]
            return dict(kb), docs

    def save_batch(self, kb_id, expected_revision, records, allow_replace=False):
        """records 经过解析；仅在所有记录和总量校验通过后提交。"""
        if not records:
            return
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            kb = db.execute('SELECT * FROM knowledge_bases WHERE id=?', (kb_id,)).fetchone()
            if kb is None or kb['revision'] != expected_revision:
                raise ValueError('知识库已在其他窗口更新，本次未保存。请刷新后重试')
            existing = {r['name_key']: dict(r) for r in db.execute('SELECT * FROM documents WHERE kb_id=?', (kb_id,))}
            keys = [r['name'].casefold() for r in records]
            if len(keys) != len(set(keys)):
                raise ValueError('本次上传有重复文件名，请先重命名')
            for record in records:
                key = record['name'].casefold()
                old = existing.get(key)
                if old and not allow_replace:
                    raise ValueError('同名文档需要勾选允许替换')
                if len(record['data']) > MAX_BYTES:
                    raise ValueError('单文件不能超过 50 MB')
                vectors = np.asarray(record['vectors'], dtype='<f4')
                chunks = record['chunks']
                if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not len(chunks) or vectors.shape[1] == 0:
                    raise ValueError('向量与片段数量不一致')
                if not np.isfinite(vectors).all():
                    raise ValueError('向量中包含无效数值')
                others = [v['dimension'] for k, v in existing.items() if k != key]
                if others and any(d != vectors.shape[1] for d in others):
                    raise ValueError('向量维度不一致，请使用原模型或创建新知识库')
                doc_id = old['id'] if old else uuid.uuid4().hex
                db.execute('''INSERT INTO documents
                    (id,kb_id,name,name_key,digest,pipeline,original,size,chunks,vectors,rows,dimension,notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(kb_id,name_key) DO UPDATE SET
                    name=excluded.name,digest=excluded.digest,pipeline=excluded.pipeline,
                    original=excluded.original,size=excluded.size,chunks=excluded.chunks,
                    vectors=excluded.vectors,rows=excluded.rows,dimension=excluded.dimension,
                    notes=excluded.notes,updated=CURRENT_TIMESTAMP''',
                    (doc_id,kb_id,record['name'],key,hashlib.sha256(record['data']).hexdigest(),record['pipeline'],
                     record['data'],len(record['data']),
                     json.dumps([{'source':record['name'],'location':c.location,'text':c.text} for c in chunks], ensure_ascii=False),
                     vectors.tobytes(),len(chunks),vectors.shape[1],json.dumps(record['notes'],ensure_ascii=False)))
                existing[key] = {'dimension': vectors.shape[1], 'id': doc_id}
            totals = db.execute('SELECT COUNT(*),COALESCE(SUM(size),0),COALESCE(SUM(rows),0) FROM documents WHERE kb_id=?', (kb_id,)).fetchone()
            if totals[0] > 10 or totals[1] > MAX_BYTES or totals[2] > 10000:
                raise ValueError('每个知识库最多 10 个文件、合计 50 MB、10000 个片段。本次未保存，请减少资料')
            db.execute('UPDATE knowledge_bases SET revision=revision+1 WHERE id=?', (kb_id,))

    def load_index(self, kb_id, expected_revision):
        with self.connect() as db:
            db.execute('BEGIN')
            kb = db.execute('SELECT revision FROM knowledge_bases WHERE id=?', (kb_id,)).fetchone()
            if kb is None or kb['revision'] != expected_revision:
                raise ValueError('知识库已更新，请刷新页面')
            rows = db.execute('SELECT chunks,vectors,rows,dimension FROM documents WHERE kb_id=? ORDER BY name', (kb_id,)).fetchall()
        chunks, vectors = [], []
        for row in rows:
            chunks.extend(Chunk(**c) for c in json.loads(row['chunks']))
            vectors.append(np.frombuffer(row['vectors'], dtype='<f4').reshape(row['rows'], row['dimension']).copy())
        return chunks, np.concatenate(vectors) if vectors else np.empty((0, 0), dtype=np.float32)

    def document(self, kb_id, doc_id):
        with self.connect() as db:
            row = db.execute('SELECT name,original,chunks,notes FROM documents WHERE kb_id=? AND id=?', (kb_id,doc_id)).fetchone()
            if row is None:
                raise ValueError('文档已被删除，请刷新页面')
            return dict(row)

    def delete_document(self, kb_id, doc_id, expected_revision):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            kb = db.execute('SELECT revision FROM knowledge_bases WHERE id=?', (kb_id,)).fetchone()
            if kb is None or kb['revision'] != expected_revision:
                raise ValueError('知识库已更新，请刷新后重试删除')
            result = db.execute('DELETE FROM documents WHERE kb_id=? AND id=?', (kb_id,doc_id))
            if not result.rowcount:
                raise ValueError('文档不存在')
            db.execute('UPDATE knowledge_bases SET revision=revision+1 WHERE id=?', (kb_id,))

    def backup(self, target):
        """SQLite 在线快照，包含原文件、向量、全部知识库，不含 API 密钥。"""
        with self.connect() as source:
            with sqlite3.connect(target) as destination:
                source.backup(destination)
