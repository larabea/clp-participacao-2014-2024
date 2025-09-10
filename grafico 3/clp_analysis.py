from __future__ import annotations
import io, re, math, unicodedata
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from dateutil import parser as dparser
from matplotlib.patches import Patch
from collections import Counter  

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

CSV_FILES = [
    DATA / "REQ 2014.csv",
    DATA / "REQ 2015.csv",
    DATA / "REQ 2016.csv",
    DATA / "REQ 2017.csv",
    DATA / "REQ 2018.csv",
    DATA / "REQ 2019.csv",
    DATA / "REQ 2021.csv",
    DATA / "REQ 2022.csv",
    DATA / "REQ 2023.csv",
    DATA / "REQ 2024.csv",
]

ANALISE_INICIO = 2014
ANALISE_FIM    = 2024
LIMIAR_OUTROS  = 7  

mpl.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300,
    "font.size": 12, "axes.titlesize": 18, "axes.labelsize": 13, "legend.fontsize": 11,
    "font.family": "Times New Roman",
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#D8D8D8", "grid.linewidth": 0.8, "grid.alpha": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3A3A3A",
    "xtick.color": "#2A2A2A", "ytick.color": "#2A2A2A",
    "axes.titlepad": 12, "axes.labelpad": 8,
    "legend.frameon": False,
    "figure.constrained_layout.use": False,
})

PARTY_COLORS = {
    "PT":  "#CC0000",
    "PSOL":"#FFD700",
    "PDT": "#0033A0",
    "PCDOB":"#8B0000",
    "PSB": "#E03C31",
    "PV":  "#006400",
    "SOLIDARIEDADE":"#F37021",
    "PL":  "#0D47A1",
    "PR":  "#003399",
    "PPS": "#FF8C42",
    "PMDB":"#228B22",
    "PSC": "#004225",
    "PRB": "#00AEEF",
    "PTB": "#B22222",
    "REPUBLICANOS": "#0066CC",
    "CIDADANIA": "#FF6600",
    "NOVO": "#000000",
    "PATRIOTA": "#009933",
    "AVANTE": "#FF9999",
    "AGIR": "#663366",
    "PODE": "#FFCC00",
    "UNIAO": "#3366CC",
    "Outros":"#B8B8B8"
}

def party_colors(labels):
    return [PARTY_COLORS.get(l, "#999999") for l in labels]

def safe_text(x) -> str:
    if x is None: return ""
    if isinstance(x, float):
        try:
            if math.isnan(x): return ""
        except Exception:
            pass
    return str(x)

def strip_quotes(s: str) -> str:
    return safe_text(s).strip().strip('"').strip("'").strip()

def deburr(s: str) -> str:
    s = safe_text(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def parse_date_any(x):
    s = safe_text(x)
    if not s: return pd.NaT
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        try:
            dt = pd.to_datetime(dparser.parse(s, dayfirst=True, fuzzy=True))
        except Exception:
            return pd.NaT
    return dt

def read_csv_tolerante(path: Path) -> pd.DataFrame:
    try:
        raw = path.read_text(encoding="utf-8-sig", errors="ignore")
        lines = raw.splitlines()
        header_idx = None
        pat = re.compile(r'^\s*"?Proposições"?\s*;', flags=re.IGNORECASE)
        for i, ln in enumerate(lines):
            if pat.search(ln):
                header_idx = i
                break
        if header_idx is None:
            for i, ln in enumerate(lines):
                if "Proposições" in ln and "Ementa" in ln and "Partido" in ln:
                    header_idx = i
                    break
            if header_idx is None:
                header_idx = 3

        buf = "\n".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(buf), sep=";", encoding="utf-8-sig",
                         engine="python", dtype=str, header=0)
        df.columns = [deburr(strip_quotes(c)).lower().strip() for c in df.columns]

        ren = {
            "proposicoes": "proposicoes",
            "ementa": "ementa",
            "apresentacao": "data",
            "partido": "partido",
            "autor": "autor",
            "uf": "uf",
        }
        for k, v in ren.items():
            if k in df.columns and v != k:
                df = df.rename(columns={k: v})

        for c in ("proposicoes", "ementa", "data", "partido"):
            if c not in df.columns:
                df[c] = ""

        return df
    except Exception as e:
        print(f"Erro ao ler arquivo {path}: {e}")
        return pd.DataFrame(columns=["proposicoes", "ementa", "data", "partido"])

def load_all_reqs() -> pd.DataFrame:
    frames = []
    for f in CSV_FILES:
        if f.exists():
            print(f"Processando {f.name}...")
            df = read_csv_tolerante(f)
            if not df.empty:
                tipo = df["proposicoes"].map(lambda s: safe_text(s).upper())
                df = df[tipo.str.startswith("REQ", na=False)].copy()
                ano_match = re.search(r'REQ\s*(\d{4})\.csv', f.name)
                if ano_match:
                    df["ano"] = int(ano_match.group(1))
                else:
                    df["data"] = df["data"].map(strip_quotes).apply(parse_date_any)
                    df["ano"] = df["data"].dt.year
                frames.append(df)

    if not frames:
        raise FileNotFoundError("Nenhum arquivo 'REQ ####.csv' encontrado em ./data")

    base = pd.concat(frames, ignore_index=True)
    base = base[(base["ano"] >= ANALISE_INICIO) & (base["ano"] <= ANALISE_FIM)].copy()
    return base

def norm_tokens(partidos: str) -> list[str]:
    s = deburr(safe_text(partidos)).upper()
    if not s.strip():
        return []

    parts = re.split(r"[;,/]+|\s{2,}", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"[^A-ZÁÂÃÀÉÊÍÓÔÕÚÇÜ ]", "", p).strip()
        if p in ("UNIAO BRASIL", "UNIAO", "UNIÃO", "UNIÃO BRASIL"):
            p = "UNIAO"
        if p in ("PC DO B", "PCDOB", "PCD0B"):
            p = "PCDOB"
        if p.startswith("REPUBLIC"):
            p = "REPUBLICANOS"
        if p.startswith("SOLIDAR"):
            p = "SOLIDARIEDADE"
        if p == "SOLIDARI":
            p = "SOLIDARIEDADE"
        if p == "PSOL23":
            p = "PSOL"
        if p == "REDE SUSTENTABILIDADE":
            p = "REDE"
        out.append(p)
    return [x for x in out if x]

SEM_SIGLA = {"AGIR","AVANTE","CIDADANIA","NOVO","PATRIOTA","REDE","REPUBLICANOS","SOLIDARIEDADE","UNIAO"}

SIGLA_TO_NOME = {
    "PT": "Partido dos Trabalhadores",
    "PSOL": "Partido Socialismo e Liberdade",
    "PDT": "Partido Democrático Trabalhista",
    "PCDOB": "Partido Comunista do Brasil",
    "PSB": "Partido Socialista Brasileiro",
    "PV": "Partido Verde",
    "PL": "Partido Liberal",
    "PPS": "Partido Popular Socialista",
    "PR": "Partido da República",
    "PMDB": "Movimento Democrático Brasileiro",
    "PSC": "Partido Social Cristão",
    "PRB": "Partido Republicano Brasileiro",
    "PTB": "Partido Trabalhista Brasileiro",
    "REPUBLICANOS": "Republicanos",
    "CIDADANIA": "Cidadania",
    "NOVO": "Partido Novo",
    "PATRIOTA": "Patriota",
    "AVANTE": "Avante",
    "AGIR": "Agir",
    "PODE": "Podemos",
    "UNIAO": "União Brasil",
    "SOLIDARIEDADE": "Solidariedade"
}

NOME_SEM_SIGLA = {
    "AGIR": "Agir",
    "AVANTE": "Avante",
    "CIDADANIA": "Cidadania",
    "NOVO": "Partido Novo",
    "PATRIOTA": "Patriota",
    "REDE": "Rede Sustentabilidade",
    "REPUBLICANOS": "Republicanos",
    "SOLIDARIEDADE": "Solidariedade",
    "UNIAO": "União Brasil",
}

print("Carregando dados de 2014 a 2024...")
df = load_all_reqs()
print(f"Total de requerimentos encontrados: {len(df)}")

combo_counter = Counter()

all_parties = []
for _, row in df.iterrows():
    parties = norm_tokens(row.get("partido", ""))
    all_parties.extend(parties)  
    uniq = sorted(set(parties))
    if len(uniq) >= 2:
        combo = "/".join(uniq)
        combo_counter[combo] += 1

cont_raw = pd.Series(all_parties, name="token").value_counts()
print(f"Total de contagens (incluindo múltiplos por requerimento): {cont_raw.sum()}")

mask_outros = cont_raw < LIMIAR_OUTROS
outros_total = int(cont_raw[mask_outros].sum())
cont = cont_raw[~mask_outros].copy()
if outros_total > 0:
    cont["Outros"] = outros_total

cont = cont.sort_values(ascending=False)
total = int(cont.sum())
perc = (cont / total * 100).round(1)

print(f"\nTotal de requerimentos por partido (2014-2024):")
for partido, quantidade in cont.items():
    print(f"{partido}: {quantidade} ({perc[partido]}%)")

if combo_counter:
    total_conjuntos = sum(combo_counter.values())
    print(f"\nRequerimentos feitos de forma conjunta (apenas combinações) — total: {total_conjuntos}")
    for combo, qtd in sorted(combo_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{combo}: {qtd} requerimentos")
else:
    print("\nRequerimentos feitos de forma conjunta (apenas combinações): 0")

def eixo_label(token: str) -> str:
    if token == "Outros": return "Outros"
    if token in SEM_SIGLA: return NOME_SEM_SIGLA.get(token, token.title())
    return token

y_labels = [eixo_label(tok) for tok in cont.index]

fig, ax = plt.subplots(figsize=(15.5, 11.0))

labels_for_color = cont.index.tolist()
colors = party_colors(labels_for_color)

bars = ax.barh(y_labels, cont.values, color=colors, linewidth=1.0, edgecolor="#2A2A2A")

ax.set_title("Gráfico 3 — Requerimentos por Partido (2014–2024)")
ax.set_xlabel("Requerimentos (n)")
ax.set_ylabel("Partido")

ax.invert_yaxis()

ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
xmax = cont.values.max() if len(cont) else 1
step = 200 if xmax >= 1000 else (100 if xmax >= 300 else 50)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
ax.grid(axis="x"); ax.grid(axis="y", visible=False)

pad = max(xmax * 0.01, 8)
for rect, n, p in zip(bars, cont.values, perc.values):
    ax.text(rect.get_x() + rect.get_width() + pad,
            rect.get_y() + rect.get_height() / 2,
            f"{int(n)} ({p:.1f}%)",
            va="center", ha="left", fontsize=11, color="#1B1B1B")

ax.set_xlim(0, xmax * 1.22)

legend_items = []
for tok, color in zip(cont.index.tolist(), colors):
    if tok == "Outros":
        legend_items.append(Patch(facecolor=color, edgecolor="#2A2A2A",
                                  label="Outros < 7 — Partidos com menos de 7 requerimentos"))
    elif tok in SEM_SIGLA:
        legend_items.append(Patch(facecolor=color, edgecolor="#2A2A2A",
                                  label=NOME_SEM_SIGLA.get(tok, tok.title())))
    else:
        legend_items.append(Patch(facecolor=color, edgecolor="#2A2A2A",
                                  label=f"{tok} — {SIGLA_TO_NOME.get(tok, tok)}"))

ncol = 3 if len(legend_items) > 16 else 2
leg = ax.legend(handles=legend_items, ncol=ncol,
                loc="upper center", bbox_to_anchor=(0.5, -0.11))
for t in leg.get_texts(): t.set_fontsize(11)

fig.subplots_adjust(bottom=0.30)

fig.text(0.5, 0.008,
         "Fonte: Elaboração própria a partir de dados da CLP (2014–2024).",
         ha="center", va="bottom", fontsize=11, color="#666666")

(FIGS / "G3_requerimentos_por_partido.png").unlink(missing_ok=True)
(FIGS / "G3_requerimentos_por_partido.svg").unlink(missing_ok=True)
plt.savefig(FIGS / "G3_requerimentos_por_partido.png")
plt.savefig(FIGS / "G3_requerimentos_por_partido.svg")
plt.close()

print("\nGráfico salvo em 'figs/G3_requerimentos_por_partido.png'")