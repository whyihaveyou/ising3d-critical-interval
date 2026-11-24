import re,fitz
aux=open('main.aux').read()
num2key={int(n):k for k,n in re.findall(r'\\bibcite\{([^}]*)\}\{(\d+)\}',aux)}
d=fitz.open('main.pdf'); t=''.join(p.get_text() for p in d); body=t[:t.rfind('References')]
seen=[]
for m in re.finditer(r'\[(\d+(?:\s*[,–-]\s*\d+)*)(?:,\s*[A-Z][^\]]*)?\]', body):
    for x in re.split(r'[,–-]',m.group(1)):
        x=x.strip()
        if x.isdigit() and int(x) not in seen: seen.append(int(x))
import sys
if sys.argv[1]=='check': print("monotone:", seen==sorted(seen), len(seen)); sys.exit()
order=[num2key[n] for n in seen]
p='sections/backmatter.tex'; s=open(p).read()
start=s.index("\\bibitem{"); end=s.index("\\end{thebibliography}")
items=[x for x in re.split(r'(?=\\bibitem\{)', s[start:end]) if x.strip()]
dd={re.match(r'\\bibitem\{([^}]*)\}',it).group(1):it.rstrip()+"\n\n" for it in items}
assert set(order)==set(dd), set(order)^set(dd)
open(p,'w').write(s[:start]+"".join(dd[k] for k in order)+s[end:]); print("reordered")
