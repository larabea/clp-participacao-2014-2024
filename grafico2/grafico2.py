from __future__ import annotations
import re
import io
import unicodedata
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

ANALISE_INICIO = 2014
ANALISE_FIM = 2024
ANOS_INDEX = list(range(ANALISE_INICIO, ANALISE_FIM + 1))

mpl.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300,
    "font.size": 12, "axes.titlesize": 18, "axes.labelsize": 13, "legend.fontsize": 12,
    "font.family": "Times New Roman", "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#D0D0D0", "grid.linewidth": 0.8, "grid.alpha": 0.45,
    "axes.spines.top": False, "axes.spines.right": False,
})

COR_SUG = "#1B5E20"
COR_REQ = "#0B3D91"


def _deburr(s: str) -> str:
    s = s or ""
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _detectar_header(texto: str) -> int:
    linhas = texto.splitlines()
    for i, ln in enumerate(linhas):
        if (';' in ln or ',' in ln) and re.search(r'propos[ií]coes', _deburr(ln), flags=re.I):
            return i
    for i, ln in enumerate(linhas):
        if ';' in ln or ',' in ln:
            return i
    return 0


def _ler_csv(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    i_header = _detectar_header(raw)
    conteudo = "\n".join(raw.splitlines()[i_header:])
    header_line = conteudo.splitlines()[0]
    sep = ';' if header_line.count(';') >= header_line.count(',') else ','
    df = pd.read_csv(io.StringIO(conteudo), sep=sep,
                     dtype=str, engine="python", quotechar='"')
    return df


def _col_proposicoes(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if re.fullmatch(r'propos[ií]coes', _deburr(str(c)).lower()):
            return c
    return df.columns[0] if len(df.columns) else None


def contar_tipo(path: Path, tipo: str) -> int:
    try:
        if not path.exists():
            return 0
        df = _ler_csv(path)
        col = _col_proposicoes(df)
        if col is None:
            return 0
        serie = df[col].astype(str).str.strip()
        mask_header_rep = serie.str.lower().eq(str(col).strip().lower())
        re_inicio = re.compile(rf'^\s*{tipo}\b', flags=re.I)
        mask_tipo = serie.str.match(re_inicio)
        return int((mask_tipo & ~mask_header_rep).sum())
    except Exception as e:
        print(f"[WARN] Falha lendo {path.name}: {e}")
        return 0


def extrair_ano(nome_arquivo: str) -> int | None:
    m = re.search(r'(\d{4})', nome_arquivo)
    return int(m.group(1)) if m else None


def carregar_dados_anuais() -> dict[int, dict[str, int]]:
    dados_por_ano = {ano: {"SUG": 0, "REQ": 0} for ano in ANOS_INDEX}
    padrao_req = re.compile(r'^REQ[ _]\d{4}\.csv$', re.I)
    padrao_sug = re.compile(r'^SUG[ _]\d{4}\.csv$', re.I)
    for path in DATA.glob("*.csv"):
        ano = extrair_ano(path.name)
        if ano is None or ano not in dados_por_ano:
            continue
        if padrao_req.match(path.name):
            dados_por_ano[ano]["REQ"] = contar_tipo(path, "REQ")
            print(f"{path.name}: {dados_por_ano[ano]['REQ']} requerimentos")
        elif padrao_sug.match(path.name):
            dados_por_ano[ano]["SUG"] = contar_tipo(path, "SUG")
            print(f"{path.name}: {dados_por_ano[ano]['SUG']} sugestões")
    return dados_por_ano


print("Carregando dados de 2014 a 2024...")
dados_anuais = carregar_dados_anuais()

anos = ANOS_INDEX
sugestoes = [dados_anuais[a]["SUG"] for a in anos]
requerimentos = [dados_anuais[a]["REQ"] for a in anos]

print("\nResumo por ano:")
print("Ano\tSUG\tREQ")
for a in anos:
    print(f"{a}\t{dados_anuais[a]['SUG']}\t{dados_anuais[a]['REQ']}")

fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=False)
ax.plot(anos, sugestoes, marker='o', linewidth=2.5,
        markersize=8, color=COR_SUG, label='Sugestões (SUG)')
ax.plot(anos, requerimentos, marker='s', linewidth=2.5,
        markersize=8, color=COR_REQ, label='Requerimentos (REQ)')

ax.set_ylim(0, 210)

ax.yaxis.grid(True)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ymin, ymax = ax.get_ylim()
dy = (ymax - ymin) * 0.02
gap = (ymax - ymin) * 0.015

for x, s, r in zip(anos, sugestoes, requerimentos):
    # posicionamento padrão
    if s > 0:
        y_s = s - dy
        if s <= 5:
            y_s = s + (dy * 2.5)
        if r > 0 and abs(s - r) < (ymax - ymin) * 0.12:
            y_s = min(y_s, r - (gap + dy))

    if r > 0:
        y_r = r + dy
        if s > 0 and abs(s - r) < (ymax - ymin) * 0.12:
            y_r = max(y_r, s + (gap + dy))

    # ==== CORREÇÃO PONTUAL DOS RÓTULOS DE 2014 (trocar 31 ↔ 45) ====
    if x == 2014:
        # mantém as posições corretas (perto dos pontos), mas troca apenas os textos
        if s > 0:
            ax.text(x, y_s, str(r), ha='center', va='top',
                    fontsize=10, color=COR_SUG, fontweight='bold',
                    clip_on=False, zorder=5)
        if r > 0:
            ax.text(x, y_r, str(s), ha='center', va='bottom',
                    fontsize=10, color=COR_REQ, fontweight='bold',
                    clip_on=False, zorder=5)
        continue
    # =================================================================

    if s > 0:
        ax.text(x, y_s, str(s), ha='center', va='top',
                fontsize=10, color=COR_SUG, fontweight='bold',
                clip_on=False, zorder=5)

    if r > 0:
        ax.text(x, y_r, str(r), ha='center', va='bottom',
                fontsize=10, color=COR_REQ, fontweight='bold',
                clip_on=False, zorder=5)

ax.set_title("Gráfico 2 – Evolução histórica de Sugestões (SUG) e Requerimentos (REQ), 2014–2024",
             pad=70, fontsize=16, fontweight='bold')

ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade")
ax.set_xticks(anos)
ax.set_xticklabels(anos, rotation=45)

ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.20),
          borderaxespad=1.2, frameon=True)

fig.subplots_adjust(bottom=0.16, top=0.88, left=0.07, right=0.98)
fig.text(0.5, 0.035, "Fonte: Elaboração própria a partir de dados da CLP (2014–2024).",
         ha='center', va='bottom', fontsize=10, color='#666666')

for nome in ("G2_evolucao_sugestoes_requerimentos.png", "G2_evolucao_sugestoes_requerimentos.svg"):
    (FIGS / nome).unlink(missing_ok=True)
    plt.savefig(FIGS / nome, bbox_inches='tight')
plt.close()

print("\nGráfico salvo em 'figs/G2_evolucao_sugestoes_requerimentos.(png|svg)'")
