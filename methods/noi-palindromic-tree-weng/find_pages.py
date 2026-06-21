import pdfplumber
p='/srv/home/bohanlyu/innovation_proior/methods/noi-palindromic-tree-weng/refs/lunwenji2017.pdf'
with pdfplumber.open(p) as pdf:
    print('total pages', len(pdf.pages))
    for i,pg in enumerate(pdf.pages):
        t = pg.extract_text() or ''
        if '回文树' in t or '翁文涛' in t or '回文自动机' in t:
            for ln in t.split('\n'):
                if '回文树' in ln or '翁文涛' in ln or '回文自动机' in ln:
                    print(i+1, repr(ln[:70]))
                    break
