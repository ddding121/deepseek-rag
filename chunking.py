"""在来源边界内按段落、句子打包；长句才硬切，标题以原文前缀保留。"""
import re


def heading(text):
    text=text.strip()
    return len(text)<=60 and bool(re.match(r'^(#{1,6}\s|第[一二三四五六七八九十百0-9]+[章节篇]|[一二三四五六七八九十]+[、.]|\d+(?:\.\d+)*[、.\s])',text))


def split_semantic(text,size=350,overlap=50):
    if size<100 or not 0<=overlap<size:
        raise ValueError('无效切块参数')
    lines=[re.sub(r'[ \t]+',' ',line).strip() for line in text.splitlines() if line.strip()]
    title=''
    buffer=''
    def prefix():
        return title+'\n' if title else ''
    for line in lines:
        if heading(line):
            if buffer:
                yield prefix()+buffer
                buffer=''
            elif title:
                yield title
            title=line
            continue
        for sentence in re.split(r'(?<=[。！？!?；;])',line):
            sentence=sentence.strip()
            if not sentence:
                continue
            capacity=size-len(prefix())
            if len(sentence)>capacity:
                if buffer:
                    yield prefix()+buffer
                    buffer=''
                step=capacity-overlap
                for start in range(0,len(sentence),step):
                    piece=sentence[start:start+capacity]
                    yield prefix()+piece
                    if start+capacity>=len(sentence):
                        break
                continue
            if buffer and len(buffer)+1+len(sentence)>capacity:
                yield prefix()+buffer
                # 仅保留末尾完整短句作为重叠，不硬截断一句话。
                tail=re.split(r'(?<=[。！？!?；;])',buffer)
                tail=next((x.strip() for x in reversed(tail) if x.strip()),'')
                buffer=tail if len(tail)<=overlap and len(tail)+1+len(sentence)<=capacity else ''
            buffer=(buffer+'\n'+sentence).strip()
    if buffer:
        yield prefix()+buffer
    elif title:
        # 只有标题的段落也应保存。
        yield title


def group_word_sections(sections):
    """合并连续 Word 正文段落；不跨表格、图片或 PDF 页合并。"""
    pending=[]
    for loc,text in sections:
        if re.fullmatch(r'段落 \d+',loc):
            pending.append((loc,text))
            continue
        if pending:
            yield (pending[0][0] if len(pending)==1 else pending[0][0]+' 至 '+pending[-1][0]), '\n'.join(t for _,t in pending)
            pending=[]
        yield loc,text
    if pending:
        yield (pending[0][0] if len(pending)==1 else pending[0][0]+' 至 '+pending[-1][0]), '\n'.join(t for _,t in pending)
