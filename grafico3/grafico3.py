from __future__ import annotations
from pathlib import Path
import re, unicodedata
import pandas as pd, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE if HERE.name.lower() == "figs" else (HERE / "figs")
FIGS.mkdir(parents=True, exist_ok=True)
DATA = HERE / "data" if (HERE / "data").exists() else HERE.parent / "data"
ARQ = DATA / "compos.csv"
if not ARQ.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {ARQ}")

mpl.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300,
    "font.size": 12, "axes.titlesize": 18, "axes.labelsize": 13,
    "font.family": "Times New Roman",
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#E3E3E3", "grid.linewidth": 0.9, "grid.alpha": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
})

NOMES_COMPLETOS = {
    "PT":"PT — Partido dos Trabalhadores","PSOL":"PSOL — Partido Socialismo e Liberdade",
    "PDT":"PDT — Partido Democrático Trabalhista","PSB":"PSB — Partido Socialista Brasileiro",
    "PL":"PL — Partido Liberal","PR":"PR — Partido da República","PRD":"PRD — Partido da Renovação Democrática",
    "UNIÃO":"UNIÃO — União Brasil",   
    "SOLIDARIEDADE":"Solidariedade","SD":"Solidariedade",
    "REPUBLICANOS":"Republicanos","PRB":"PRB — Partido Republicano Brasileiro",
    "PSD":"PSD — Partido Social Democrático","DEM":"DEM — Democratas",
    "MDB":"MDB — Movimento Democrático Brasileiro","PMDB":"MDB — Movimento Democrático Brasileiro",
    "PTB":"PTB — Partido Trabalhista Brasileiro","PCDOB":"PCdoB — Partido Comunista do Brasil",
    "PCdoB":"PCdoB — Partido Comunista do Brasil","PV":"PV — Partido Verde","PSC":"PSC — Partido Social Cristão",
    "PSDB":"PSDB — Partido da Social Democracia Brasileira","PROS":"PROS — Partido Republicano da Ordem Social",
    "PATRIOTA":"Patriota","PMB":"PMB — Partido da Mulher Brasileira","PTC":"PTC — Partido Trabalhista Cristão",
    "REDE":"REDE Sustentabilidade","PP":"PP — Partido Progressista","PSL":"PSL — Partido Social Liberal",
}


PALETA_CONTRASTE = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffd92f","#a65628","#f781bf",
    "#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#e5c494","#1b9e77","#d95f02",
    "#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666","#1f78b4","#33a02c",
    "#fb9a99","#b2df8a","#cab2d6","#fdbf6f","#ffff99","#6a3d9a","#b15928"
]


CORES_FIXAS = {
    "PT": "#E41A1C","PSOL": "#FFD400","PCdoB": "#A50F15","PCDOB": "#A50F15",
    "PDT": "#08519C","PSB": "#FFB300","PV": "#1A9850","REDE": "#FF7F00",
    "MDB": "#006837","PSD": "#41AB5D","PSDB": "#3182BD","SD": "#FB9A29",
    "UNIÃO": "#253494","PATRIOTA": "#006D2C","PL": "#1E2A78","PP": "#4AA5FF",
    "PR": "#005DAA","PRB": "#005DAA","REPUBLICANOS": "#005DAA","PSC": "#008000",
    "PROS": "#F58220","PTB": "#E41A1C","PMB": "#FF69B4","PRD": "#A6CE39","PSL": "#FFE34F"
}

def deburr(s:str)->str:
    if not s: return ""
    return "".join(ch for ch in unicodedata.normalize("NFKD",str(s)) if not unicodedata.combining(ch))

def limpar_sigla(cell:str)->str:
    if not isinstance(cell,str): return ""
    s = deburr(cell.strip().upper()).replace("\\","/")
    sigla = s.split("/")[0].strip()
    sigla = (sigla
             .replace("PCDO B","PCDOB")
             .replace("PC DO B","PCDOB")
             .replace("PCD0B","PCDOB")
             .replace("UNIAO BRASIL","UNIÃO")
             .replace("UNIÃO BRASIL","UNIÃO")
             .replace("UNIO BRASIL","UNIÃO")
             .replace("UNI�O BRASIL","UNIÃO")
             .replace("UNIAO","UNIÃO")
             .replace("UNI�O","UNIÃO")
             .replace("UNIIO","UNIÃO")
             .replace("UNIO","UNIÃO")
             .replace("SOLIDARIEDADE","SD")
    )
    import re as _re
    return _re.sub(r"[^A-ZÇÃÕ]", "", sigla)

def hex_luminance(hexcolor:str)->float:
    r,g,b = (int(hexcolor[i:i+2],16)/255 for i in (1,3,5))
    return 0.2126*r + 0.7152*g + 0.0722*b
def label_color_for(bg_hex:str)->str:
    return "black" if hex_luminance(bg_hex)>0.6 else "white"


for enc in ("utf-8-sig","latin-1","cp1252"):
    try:
        df = pd.read_csv(ARQ, sep=";", dtype=str, encoding=enc, engine="python"); break
    except UnicodeDecodeError: continue

anos = [str(a) for a in range(2014,2025)]
dados = []
for ano in anos:
    if ano in df.columns:
        for p in df[ano].dropna():
            if str(p).strip():
                s = limpar_sigla(p)
                if s: dados.append({"Ano":int(ano),"Sigla":s})
df_limpo = pd.DataFrame(dados)

contagem = df_limpo.groupby(["Ano","Sigla"]).size().reset_index(name="Cadeiras")
tabela = contagem.pivot(index="Ano", columns="Sigla", values="Cadeiras").fillna(0)
tabela = tabela.reindex(range(2014,2025), fill_value=0)
tabela = tabela.loc[:, tabela.sum().sort_values(ascending=False).index]

totais_por_partido = tabela.sum(axis=0).sort_values(ascending=False)
totais_por_ano = tabela.sum(axis=1)

colunas = list(tabela.columns)
cores_cols = {sig: CORES_FIXAS.get(sig, PALETA_CONTRASTE[i % len(PALETA_CONTRASTE)])
              for i, sig in enumerate(colunas)}

fig, ax = plt.subplots(figsize=(16,11))
bottom = np.zeros(len(tabela), dtype=float)
x = np.arange(len(tabela.index))

handles, labels = [], []
for sig in colunas:
    vals = tabela[sig].to_numpy(float)
    cor = cores_cols[sig]
    bar = ax.bar(x, vals, bottom=bottom, color=cor, edgecolor="white", linewidth=0.6)
    for xi, v, b in zip(x, vals, bottom):
        if v >= 1:
            ax.text(
                xi, b + v/2, f"{sig}\n{int(v)}",
                ha="center", va="center",
                fontsize=8, color=label_color_for(cor), fontweight="bold", linespacing=1.2
            )
    bottom += vals
    handles.append(bar[0]); labels.append(NOMES_COMPLETOS.get(sig, sig))

ax.set_title("Gráfico 3 — Composição partidária da CLP (2014–2024)", pad=18, fontweight="bold")
ax.set_xlabel("Ano"); ax.set_ylabel("Cadeiras (nº de membros)")
ax.set_xticks(x); ax.set_xticklabels([str(a) for a in tabela.index])
ax.yaxis.grid(True, which="major", alpha=0.4); ax.xaxis.grid(False)

for xi, ano in enumerate(tabela.index):
    total_ano = int(totais_por_ano.loc[ano])
    if total_ano > 0:
        ax.text(xi, bottom[xi] + 0.8, str(total_ano),
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333333")

leg = ax.legend(handles, labels, loc="upper center",
    bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False,
    fontsize=10, labelspacing=0.8, columnspacing=1.5,
    handlelength=1.2, handleheight=1.0, borderaxespad=0.2, borderpad=0.25)

plt.subplots_adjust(bottom=0.38)
fig.text(0.5, -0.06, "Fonte: Elaboração própria a partir dos dados da CLP (2014–2024).",
         ha="center", va="bottom", fontsize=10, color="#666666")

ax.set_ylim(0, (bottom.max() if len(bottom) else 0) * 1.18)
plt.tight_layout()

for nome in ("G3_composicao_partidaria.png", "G3_composicao_partidaria.svg"):
    (FIGS/nome).unlink(missing_ok=True)
    plt.savefig(FIGS/nome, bbox_inches="tight")
plt.close()

print("\n=== Totais por partido (2014–2024) ===")
for sig, tot in totais_por_partido.items():
    nome = NOMES_COMPLETOS.get(sig, sig)
    print(f"{sig:<12}{int(tot):>5}  —  {nome}")

print("\n=== Totais por ano ===")
for ano, tot in totais_por_ano.items():
    print(f"{int(ano)}: {int(tot)}")

print("\n=== Totais por ano, por partido ===")
for ano in tabela.index:
    serie = tabela.loc[ano]
    serie = serie[serie > 0].sort_values(ascending=False)
    if serie.empty: continue
    print(f"\n-- {int(ano)} --")
    for sig, qtd in serie.items():
        nome = NOMES_COMPLETOS.get(sig, sig)
        print(f"{sig:<12}{int(qtd):>5}  —  {nome}")

print(f"\nGráfico salvo em {FIGS}/G3_composicao_partidaria.(png|svg)")
