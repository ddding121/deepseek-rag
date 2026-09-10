import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from ingestion import ingest, pipeline_key
from rag import parse_file, answer
from retrieval import BM25, search, similarity_reminder
from storage import Store

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
st.set_page_config(page_title='DeepSeek 持久知识库', page_icon='📚', layout='wide')
store = Store(ROOT / 'data' / 'knowledge.db')


@st.cache_resource
def load_encoder(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device='cpu')


@st.cache_resource
def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(os.getenv('RERANK_MODEL', 'BAAI/bge-reranker-base'), max_length=512, device='cpu')


@st.cache_resource
def load_ocr():
    from ocr_engine import OCREngine
    return OCREngine()


st.title('📚 DeepSeek 知识库 V3')
st.caption('保存资料 · 增量更新 · 本地 OCR · 带来源的问答')
st.info('知识库自动保存在本机，重启后仍可使用。提问时，问题和相关文字片段会发送到 DeepSeek。')

with st.sidebar:
    st.header('知识库')
    kbs = store.list_kbs()
    with st.expander('新建知识库', expanded=not kbs):
        with st.form('create_kb'):
            kb_name = st.text_input('知识库名称', placeholder='例如：课程资料')
            new_model = st.text_input('中文向量模型', value=os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5'))
            create = st.form_submit_button('创建')
        if create:
            try:
                st.session_state.kb_choice = store.create_kb(kb_name, new_model)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if not kbs:
        st.info('先创建一个知识库，再上传资料。')
        st.stop()
    names = {k['id']:k['name'] for k in kbs}
    if st.session_state.get('kb_choice') not in names:
        st.session_state.kb_choice = kbs[0]['id']
    kb_id = st.selectbox('当前知识库', options=list(names), format_func=names.get, key='kb_choice')
    st.divider()
    api_key = st.text_input('DeepSeek API Key', type='password', value=os.getenv('DEEPSEEK_API_KEY',''))
    model = st.text_input('DeepSeek 模型', value=os.getenv('DEEPSEEK_MODEL','deepseek-chat'))
    top_k = st.slider('检索片段数', 1, 10, 5)
    threshold = st.slider('语义候选最低相似度', 0.0, 1.0, 0.5, 0.05)
    search_label = st.selectbox('检索方式', ['混合检索（关键词 + 语义）', '仅语义检索（对照）'])
    search_mode = 'hybrid' if search_label.startswith('混合') else 'dense'
    use_reranker = st.checkbox('启用模型重排序（首次需要下载，CPU 较慢）', value=False)
    st.caption('混合检索允许关键词命中补充低语义分片段；阈值只筛选语义候选。')
    st.caption('相似度不是答案可信度。请写完整问题，本版不做连续追问改写。')
    if st.button('刷新知识库列表'):
        st.rerun()

kb, docs = store.snapshot(kb_id)
# 变更版本后立刻丢弃旧索引与问答，防止删除后仍引用旧文档。
session_key = (kb_id, kb['revision'])
if st.session_state.get('index_key') != session_key:
    st.session_state.index_key = session_key
    st.session_state.history = []
    st.session_state.pop('active_index', None)
    st.session_state.pop('lexical_index', None)

st.subheader(kb['name'])
a,b,c = st.columns(3)
a.metric('已保存文档', len(docs))
b.metric('文字片段', sum(d['rows'] for d in docs))
c.metric('原文件合计', f"{sum(d['size'] for d in docs) / 1024 / 1024:.2f} MB")
if 'notice' in st.session_state:
    st.success(st.session_state.pop('notice'))

chat_tab, manage_tab, backup_tab = st.tabs(['知识问答', '文档管理', '备份与恢复'])
with manage_tab:
    st.caption('每个知识库最多 10 个文件、合计 50 MB、10000 个片段；每个 PDF 最多 400 页。')
    with st.expander('添加或替换文档', expanded=not docs):
        enable_ocr = st.checkbox('启用图片文字识别（OCR）', value=True)
        modes = {'自动：文字少的页面做 OCR':'auto', '全页 OCR：混合文字和图片的 PDF':'all', '仅提取 PDF 文字层':'off'}
        pdf_mode = modes[st.selectbox('PDF 识别方式', list(modes))]
        st.caption('图片和 Word 内嵌图片按 OCR 开关处理。OCR 读取文字，不理解图表趋势。')
        uploads = st.file_uploader('上传 PDF、Word 或图片', type=['pdf','docx','png','jpg','jpeg','bmp','webp'],
                                   accept_multiple_files=True, key=f'uploads_{kb_id}')
        replace = st.checkbox('允许替换同名文档（成功后旧内容将被更新）', value=False)
        st.caption('相同文件名、内容和解析设置会自动跳过。替换失败保留原内容；只处理新增或变化的文件。')
        if st.button('处理并保存到知识库', disabled=not uploads):
            if len(uploads)>10 or sum(f.size for f in uploads)>50*1024*1024:
                st.error('本次最多上传 10 个文件，合计不超过 50 MB')
            else:
                bar = st.progress(0.0, text='检查已有文档…')
                try:
                    def parse(name, data, report):
                        ocr = load_ocr() if enable_ocr else None
                        return parse_file(name,data,ocr=ocr,pdf_mode=pdf_mode,progress=report)

                    def encode(chunks, report):
                        encoder = load_encoder(kb['model'])
                        batches = []
                        for start in range(0,len(chunks),32):
                            batch = chunks[start:start+32]
                            batches.append(encoder.encode([c.text for c in batch], normalize_embeddings=True,
                                                          convert_to_numpy=True,batch_size=32))
                            report(min(start+32,len(chunks))/len(chunks))
                        return np.concatenate(batches)

                    added, skipped = ingest(store,kb_id,[(f.name,f.getvalue()) for f in uploads],
                                             pipeline_key(enable_ocr,pdf_mode),replace,parse,encode,
                                             lambda v,m:bar.progress(v,text=m))
                    st.session_state.notice = f'已保存 {added} 个文件，跳过 {len(skipped)} 个未变化文件。'
                    st.rerun()
                except Exception as exc:
                    logging.exception('保存知识库失败')
                    st.error(f'本次未保存，原知识库保持完整。{exc}' if isinstance(exc,ValueError)
                             else f'处理失败（{type(exc).__name__}），请查看 CMD 报错。原知识库保持完整。')

    if docs:
        with st.expander('升级旧知识库：用保存的原文件重建索引'):
            old_count=sum('structure-v3' not in d['pipeline'] for d in docs)
            st.write(f'当前有 {old_count} 个文档尚未采用 V3 切块。旧索引仍可问答，重建后启用新切块。')
            st.caption('使用上方 OCR 设置重新解析全部原文件；无需再次上传。扫描资料会重新做 OCR。失败保留旧索引。')
            rebuild_confirm=st.checkbox('已备份，确认重新处理当前知识库的全部文件')
            if st.button('重建当前知识库索引',disabled=not rebuild_confirm):
                bar=st.progress(0.0,text='读取保存的原文件…')
                try:
                    originals=[(d['name'],store.document(kb_id,d['id'])['original']) for d in docs]
                    def rebuild_parse(name,data,report):
                        return parse_file(name,data,ocr=load_ocr() if enable_ocr else None,
                                          pdf_mode=pdf_mode,progress=report)
                    def rebuild_encode(chunks,report):
                        encoder=load_encoder(kb['model'])
                        batches=[]
                        for start in range(0,len(chunks),32):
                            batches.append(encoder.encode([c.text for c in chunks[start:start+32]],
                                normalize_embeddings=True,convert_to_numpy=True,batch_size=32))
                            report(min(start+32,len(chunks))/len(chunks))
                        return np.concatenate(batches)
                    ingest(store,kb_id,originals,pipeline_key(enable_ocr,pdf_mode),True,
                           rebuild_parse,rebuild_encode,lambda v,m:bar.progress(v,text=m),
                           expected_revision=kb['revision'],force=True)
                    st.session_state.notice='V3 索引重建完成。'
                    st.rerun()
                except Exception as exc:
                    logging.exception('重建失败')
                    st.error(f'重建失败，旧知识库未被覆盖：{exc}' if isinstance(exc,ValueError)
                             else f'重建失败（{type(exc).__name__}），旧知识库保留。请查看 CMD 报错。')

    if docs:
        st.dataframe([{'文件名':d['name'],'大小 MB':round(d['size']/1024/1024,2),
                       '片段数':d['rows'],'更新时间 UTC':d['updated']} for d in docs])
        by_id = {d['id']:d for d in docs}
        doc_id = st.selectbox('查看或删除文档',list(by_id),format_func=lambda i:by_id[i]['name'],key=f'doc_{kb_id}')
        doc = store.document(kb_id,doc_id)
        notes = json.loads(doc['notes'])
        if notes:
            with st.expander(f'解析提醒（{len(notes)} 条）'):
                for note in notes:
                    st.warning(note)
        chunks_data = json.loads(doc['chunks'])
        with st.expander('查看已保存的文字片段'):
            page_count = (len(chunks_data)+19)//20
            page_num = st.number_input('片段列表页码',min_value=1,max_value=max(1,page_count),value=1,key=f'page_{doc_id}_{kb["revision"]}')
            for item in chunks_data[(page_num-1)*20:page_num*20]:
                st.text(item['location'])
                st.text(item['text'])
        st.download_button('下载保存的原文件',data=doc['original'],file_name=doc['name'],key=f'original_{doc_id}')
        confirm = st.checkbox(f'确认从当前知识库删除：{doc["name"]}',key=f'confirm_{doc_id}_{kb["revision"]}')
        if st.button('删除选中文档',disabled=not confirm):
            try:
                store.delete_document(kb_id,doc_id,kb['revision'])
                st.session_state.notice='文档及其检索片段已删除。'
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    else:
        st.info('当前知识库还没有文档。')

with backup_tab:
    st.write('知识库保存在项目文件夹的 data/knowledge.db，包含原文件、文字和向量。')
    st.warning('删除整个项目文件夹也会删除知识库。请将备份下载到其他文件夹或硬盘。备份包含全部知识库与原资料，请妥善保管。')
    if st.button('生成全部知识库备份'):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'knowledge-backup.db'
            store.backup(target)
            st.download_button('下载本次备份',data=target.read_bytes(),file_name='knowledge-backup.db',mime='application/octet-stream')
    st.write('恢复：先关闭程序，备份当前 data 文件夹，将备份文件复制到 data 并重命名为 knowledge.db，再启动程序。恢复会以备份内容替换当前全部知识库。')
    st.caption('API 密钥和聊天记录不写入数据库。向量模型绑定知识库；更换模型请新建知识库重新导入资料。')


def sources(hits, details=None):
    if hits:
        with st.expander('查看检索原文'):
            for i,(chunk,score) in enumerate(hits,1):
                st.text(f'[{i}] {chunk.source} · {chunk.location} · 相似度 {score:.3f}')
                if details and i <= len(details):
                    d=details[i-1]
                    st.caption(f"BM25 {d['bm25']:.3f} · 融合排序分 {d['rrf']:.4f} · 重排序分 {d['rerank'] if d['rerank'] is not None else '未启用'}")
                st.text(chunk.text)


with chat_tab:
    if not docs:
        st.info('请到“文档管理”上传并保存资料，然后开始提问。')
    else:
        st.success('已有知识库可直接提问，无需重新上传或做 OCR。')
    for item in st.session_state.history:
        with st.chat_message('user'):
            st.write(item['question'])
        with st.chat_message('assistant'):
            if item.get('similarity_notice'):
                st.warning(item['similarity_notice'])
            st.write(item['answer'])
            sources(item['hits'],item.get('details'))
    question = st.chat_input('根据当前知识库提问，请写完整问题',disabled=not docs)
    if question:
        if not api_key.strip():
            st.warning('请在左侧填写 DeepSeek API Key')
        elif len(question)>2000:
            st.warning('问题最多 2000 字')
        else:
            try:
                with st.spinner('读取已有索引并检索…'):
                    if 'active_index' not in st.session_state:
                        st.session_state.active_index = store.load_index(kb_id,kb['revision'])
                    chunks,vectors=st.session_state.active_index
                    if 'lexical_index' not in st.session_state:
                        st.session_state.lexical_index=BM25([c.text for c in chunks])
                    reranker=load_reranker() if use_reranker else None
                    query_diagnostics={}
                    hits,details=search(question,chunks,vectors,load_encoder(kb['model']),top_k,threshold,
                                        mode=search_mode,lexical=st.session_state.lexical_index,reranker=reranker,diagnostics=query_diagnostics)
                    result=answer(question,hits,api_key.strip(),model.strip())
                    # API 请求期间若另一窗口更新文档，不展示旧版本结果。
                    current,_=store.snapshot(kb_id)
                    if current['revision']!=kb['revision']:
                        raise ValueError('回答期间知识库有更新，请刷新后重新提问')
                st.session_state.history.append(dict(question=question,answer=result,hits=hits,details=details,
                                                     similarity_notice=similarity_reminder(query_diagnostics.get('max_semantic'),threshold)))
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f'问答失败（{type(exc).__name__}）。请检查模型下载、API 密钥、余额、模型名称及网络。')
