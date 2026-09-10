"""增量处理：内容与解析配置相同时跳过；批次成功后原子保存。"""
import hashlib
import json


def pipeline_key(ocr_enabled, pdf_mode):
    return json.dumps({'parser': 'structure-v3', 'ocr': bool(ocr_enabled), 'pdf_mode': pdf_mode}, sort_keys=True)


def plan_uploads(existing, uploads, pipeline, allow_replace):
    keys = [name.casefold() for name, data in uploads]
    if len(set(keys)) != len(keys):
        raise ValueError('上传列表有重复文件名，请先重命名')
    old = {d['name_key']:d for d in existing}
    pending, skipped = [], []
    for name, data in uploads:
        doc = old.get(name.casefold())
        if doc and doc['digest'] == hashlib.sha256(data).hexdigest() and doc['pipeline'] == pipeline:
            skipped.append(name)
        elif doc and not allow_replace:
            raise ValueError(f'“{name}”已有不同内容或解析设置；如需更新请勾选允许替换')
        else:
            pending.append((name, data))
    projected = {d['name_key']:d['size'] for d in existing}
    projected.update({name.casefold():len(data) for name,data in pending})
    if len(projected) > 10 or sum(projected.values()) > 50 * 1024 * 1024:
        raise ValueError('更新后知识库将超过 10 个文件或合计 50 MB，请先删除不需要的文档')
    return pending, skipped


def ingest(store, kb_id, uploads, pipeline, allow_replace, parse, encode, progress, expected_revision=None, force=False):
    kb, existing = store.snapshot(kb_id)
    if expected_revision is not None and kb['revision'] != expected_revision:
        raise ValueError('读取原文后知识库发生变化，请刷新后重新重建')
    pending, skipped = plan_uploads(existing, uploads, pipeline, allow_replace)
    if force:
        pending, skipped = uploads, []
    records = []
    for i, (name, data) in enumerate(pending):
        def report(value, message):
            progress((i + 0.7 * value) / len(pending), message)
        chunks, notes = parse(name, data, report)
        if sum(len(r['chunks']) for r in records) + len(chunks) > 10000:
            raise ValueError('本次解析超过 10000 个片段，请拆分资料')
        progress((i + 0.7) / len(pending), f'{name} · 生成向量')
        vectors = encode(chunks, lambda value: progress((i + 0.7 + 0.3 * value) / len(pending), f'{name} · 生成向量'))
        records.append(dict(name=name,data=data,pipeline=pipeline,chunks=chunks,vectors=vectors,notes=notes))
    store.save_batch(kb_id, kb['revision'], records, allow_replace)
    progress(1.0, f'已保存 {len(records)} 个文件，跳过 {len(skipped)} 个未变化文件')
    return len(records), skipped
