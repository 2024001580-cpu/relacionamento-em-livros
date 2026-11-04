import csv
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import spacy
from PIL import Image
from spacy.cli import download as baixar_modelo_spacy
from groq import Groq, RateLimitError

LINK_PADRAO = "https://www.gutenberg.org/cache/epub/2701/pg2701-images.html"
MODELO = "en_core_web_sm"
MIN_MENCOES = 5
MAX_PERSONAGENS = 30
JANELA = 3
MIN_ARESTA = 2
MIN_PALAVRAS_CAPITULO = 50

GROQ_MODELO = "openai/gpt-oss-120b"
MAX_PARES_LLM = 40
MAX_TRECHOS = 8
MAX_CHARS_TRECHO = 400
PAUSA_GROQ = 2.5
RELACOES = ("amigo", "indiferente", "inimigo")
COR_RELACAO = {"amigo": "#2a9d8f", "indiferente": "#8d99ae", "inimigo": "#d62828"}
COR_PADRAO = "#555555"

TITULOS = {
    "mr", "mrs", "miss", "ms", "dr", "captain", "capt", "sir", "lord", "lady",
    "father", "mother", "uncle", "aunt", "old", "king", "queen", "master",
    "mister", "madame", "monsieur", "saint", "st", "professor", "prof",
    "deacon", "squire", "brother", "sister", "mate", "commodore", "colonel",
    "general", "major", "young", "little", "the",
}
PESSOA = {"i", "me", "my", "myself", "mine"}
ASPAS = re.compile(r'"[^"\n]*"|"[^"\n]*$', re.M)
SKIP = {"god", "heaven", "christ", "jesus", "satan", "devil", "lord", "providence"}

def getLivro(link):
    m = re.search(r"(?:epub/|ebooks/|files/|/pg)(\d+)", link) or re.fullmatch(r"\s*(\d+)\s*", link)
    if not m:
        sys.exit("Não foi possivel achar o ID do Livro.")
    return m.group(1)

def getTexto(id):
    urls = [
        f"https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
        f"https://www.gutenberg.org/ebooks/{id}.txt.utf-8",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                print(f"Baixado: {url}")
                return r.read().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
        except Exception as e:
            print(f"Falhou {url}: {e}")
    sys.exit("Não foi possivel baixar o texto do Livro..")

def getTitulo(txt):
    m = re.search(r"^Title:\s*(.+)$", txt, re.M)
    return m.group(1).strip() if m else "livro"

def removeCabecalho(txt):
    inicio = re.search(r"^\*\*\* ?START OF[^\n]*\n", txt, re.M)
    fim = re.search(r"^\*\*\* ?END OF", txt, re.M)
    return txt[inicio.end() if inicio else 0: fim.start() if fim else len(txt)]


def juntarLinhas(txt):
    for a, b in (("_", ""), ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        txt = txt.replace(a, b)
    paragrafos = re.split(r"\n\s*\n", txt)
    return "\n\n".join(" ".join(p.split()) for p in paragrafos if p.strip())

TITULO_CAPITULO = re.compile(
    r"^[ \t]*((?:CHAPTER|Chapter|CAPÍTULO|Capítulo)\s+(?:[IVXLCDM]+|\d+)\b[^\n]*|EPILOGUE\.?|Epilogue\.?)[ \t]*$",
    re.M,
)

def separarCapitulos(txt):
    achados = list(TITULO_CAPITULO.finditer(txt))
    corpos = []
    for i, m in enumerate(achados):
        fim = achados[i + 1].start() if i + 1 < len(achados) else len(txt)
        corpos.append(txt[m.end():fim])

    curto = [len(c.strip()) < 200 for c in corpos]
    capitulos = []
    for i, m in enumerate(achados):
        e_sumario = curto[i] or (i >= 2 and curto[i - 1] and curto[i - 2])
        corpo = juntarLinhas(corpos[i])
        if not e_sumario and len(corpo.split()) >= MIN_PALAVRAS_CAPITULO:
            capitulos.append((m.group(1).strip(), corpo))

    if len(capitulos) < 2:
        print("Capitulos não encontrados, ou insuficientes. Dividindo o livro em 20 partes.")
        palavras = juntarLinhas(txt).split()
        tamanho = max(1, len(palavras) // 20)
        capitulos = [
            (f"Parte {k + 1}", " ".join(palavras[k * tamanho:(k + 1) * tamanho]))
            for k in range(20)
        ]
    return capitulos


def carregarModelo():
    try:
        nlp = spacy.load(MODELO, disable=["lemmatizer"])
    except OSError:
        print(f"Baixando modelo {MODELO}.")
        baixar_modelo_spacy(MODELO)
        nlp = spacy.load(MODELO, disable=["lemmatizer"])
    nlp.max_length = 2_000_000
    return nlp


def limparNome(txt):
    txt = re.sub(r"['’]s$", "", txt.strip())
    partes = re.findall(r"[A-Za-z][A-Za-z\-']*", txt)
    while partes and partes[0].lower() in TITULOS:
        partes.pop(0)
    if not partes:
        return None
    nome = " ".join(partes)
    if len(nome) < 3 or not nome[0].isupper() or nome.lower() in SKIP:
        return None
    return nome


def getNomes(caps, nlp, narr=None):
    resultado, textos = [], []
    for i, doc in enumerate(nlp.pipe([corpo for _, corpo in caps], batch_size=1), 1):
        entre_aspas = [m.span() for m in ASPAS.finditer(doc.text)] if narr else []
        frases, txt_frases = [], []
        for frase in doc.sents:
            nomes = [limparNome(e.text) for e in frase.ents if e.label_ == "PERSON"]
            nomes = [n for n in nomes if n]
            if narr and narr not in nomes and any(
                t.lower_ in PESSOA and not any(a <= t.idx < b for a, b in entre_aspas)
                for t in frase
            ):
                nomes.append(narr)
            frases.append(nomes)
            txt_frases.append(frase.text.strip())
        resultado.append(frases)
        textos.append(txt_frases)
        print(f"NER: {i}/{len(caps)}", end="\r")
    print()
    return resultado, textos


def getApelidos(cont):
    simples = {n for n in cont if " " not in n}
    apelidos = {}
    for n in cont:
        if " " in n:
            partes = n.split()
            for p in (partes[-1], partes[0]):
                if p in simples:
                    apelidos[n] = p
                    break
    return apelidos

def contarRelacoes(fcap, tcap, apel, pers):
    mencoes_cap, arestas_cap = [], []
    exemplos = {}
    for frases, txt_frases in zip(fcap, tcap):
        mencoes, arestas = Counter(), Counter()
        for k in range(0, len(frases), JANELA):
            cena = set()
            for nomes in frases[k:k + JANELA]:
                for n in nomes:
                    n = apel.get(n, n)
                    if n in pers:
                        mencoes[n] += 1
                        cena.add(n)
            if len(cena) >= 2:
                trecho = " ".join(txt_frases[k:k + JANELA])[:MAX_CHARS_TRECHO]
                for par in combinations(sorted(cena), 2):
                    arestas[par] += 1
                    exemplos.setdefault(par, []).append(trecho)
        mencoes_cap.append(mencoes)
        arestas_cap.append(arestas)
    return mencoes_cap, arestas_cap, exemplos


def escolherTrechos(trechos):
    if len(trechos) <= MAX_TRECHOS:
        return trechos
    passo = len(trechos) / MAX_TRECHOS
    return [trechos[int(i * passo)] for i in range(MAX_TRECHOS)]


def classificarRelacoes(arestas_total, exemplos, titulo):
    chave = os.environ.get("GROQ_API_KEY", "").strip()
    if not chave:
        print("GROQ_API_KEY não definida: relações não serão classificadas.")
        return {}

    cliente = Groq(api_key=chave)
    pares = [p for p, w in arestas_total.most_common() if w >= MIN_ARESTA][:MAX_PARES_LLM]
    resultado = {}

    for i, (a, b) in enumerate(pares, 1):
        trechos = "\n".join(f"- {t}" for t in escolherTrechos(exemplos[(a, b)]))
        prompt = (
            f'Livro: "{titulo}".\n'
            f"Personagens: {a} e {b}.\n"
            f"Trechos em que aparecem juntos:\n{trechos}\n\n"
            f"Com base nos trechos, classifique a relação de {a} com {b} em UMA categoria:\n"
            '- "amigo": aliança, afeto, lealdade, cooperação\n'
            '- "inimigo": conflito, rivalidade, hostilidade, antagonismo\n'
            '- "indiferente": relação neutra, apenas profissional/circunstancial ou sem sinais claros\n'
            'Responda SOMENTE com JSON: {"relacao": "amigo|indiferente|inimigo", '
            '"confianca": número de 0 a 1, "motivo": "uma frase curta em português"}'
        )
        for tentativa in range(4):
            try:
                resp = cliente.chat.completions.create(
                    model=GROQ_MODELO,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=400,
                )
                dados = json.loads(resp.choices[0].message.content)
                rel = str(dados.get("relacao", "")).strip().lower()
                if rel not in RELACOES:
                    rel = "indiferente"
                resultado[(a, b)] = {"relacao": rel, "motivo": str(dados.get("motivo", "")).strip()}
                break
            except RateLimitError:
                time.sleep(10 * (tentativa + 1))
            except Exception as e:
                print(f"\nErro Groq em {a} - {b}: {e}")
                break
        print(f"Groq: {i}/{len(pares)}", end="\r")
        time.sleep(PAUSA_GROQ)
    print()
    return resultado


def desenhar(G, pos, menc, novos, tit, cam, dpi, relacoes=None):
    relacoes = relacoes or {}
    fig, ax = plt.subplots(figsize=(10, 8))
    arestas = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= MIN_ARESTA]

    grupos = {}
    for u, v in arestas:
        tipo = relacoes.get(tuple(sorted((u, v))), {}).get("relacao")
        grupos.setdefault(tipo, []).append((u, v))
    for tipo, lista in grupos.items():
        larguras = [min(0.8 + G[u][v]["weight"] ** 0.5 * 0.4, 6) for u, v in lista]
        nx.draw_networkx_edges(G, pos, edgelist=lista, width=larguras,
                               edge_color=COR_RELACAO.get(tipo, COR_PADRAO), alpha=0.75, ax=ax)

    nos = list(G.nodes)
    tamanhos = [min(100 + 25 * menc[n] ** 0.5, 1500) for n in nos]
    cores = ["#457b9d" if n in novos else "#f4a261" for n in nos]
    nx.draw_networkx_nodes(G, pos, nodelist=nos, node_size=tamanhos, node_color=cores,
                           edgecolors="#333", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    if relacoes:
        legenda = [Line2D([0], [0], color=COR_RELACAO[r], lw=3, label=r.capitalize()) for r in RELACOES]
        legenda.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#457b9d",
                              markeredgecolor="#333", markersize=9, label="Personagem novo"))
        ax.legend(handles=legenda, loc="lower left", fontsize=8, framealpha=0.85)

    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.15, max(xs) + 0.15)
    ax.set_ylim(min(ys) - 0.15, max(ys) + 0.15)
    ax.set_title(tit)
    ax.axis("off")
    fig.savefig(cam, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def criarGif(pasta, cam):
    arquivos = sorted(pasta.glob("*.png"))
    if not arquivos:
        return
    imagens = [Image.open(f).convert("RGB") for f in arquivos]
    tamanho = imagens[0].size
    imagens = [im.resize(tamanho) for im in imagens]
    imagens[0].save(cam, save_all=True, append_images=imagens[1:], duration=400, loop=0)

def main():
    link = sys.argv[1] if len(sys.argv) > 1 else input("Coloque o link da pagina no Gutenberg: ").strip()
    link = link or LINK_PADRAO

    narrador = None
    if input("O livro é em primeira pessoa? (s/n): ").strip().lower() in ("s", "sim", "y", "yes"):
        nome = input("Qual o nome do personagem principal?: ").strip()
        narrador = (nome[:1].upper() + nome[1:]) or None

    id_livro = getLivro(link)
    bruto = getTexto(id_livro)
    titulo = getTitulo(bruto)
    print(f"Livro: {titulo}")
    capitulos = separarCapitulos(removeCabecalho(bruto))
    print(f"{len(capitulos)} capítulos encontrados.")

    saida = Path("output") / re.sub(r"[^A-Za-z0-9]+", "_", titulo).strip("_")[:50]
    pasta_frames = saida / "evolution"
    pasta_frames.mkdir(parents=True, exist_ok=True)

    nlp = carregarModelo()
    frases_por_capitulo, textos_por_capitulo = getNomes(capitulos, nlp, narrador)

    contagem_bruta = Counter(n for frases in frases_por_capitulo for f in frases for n in f)
    apelidos = getApelidos(contagem_bruta)
    canonicos = Counter()
    for n, c in contagem_bruta.items():
        canonicos[apelidos.get(n, n)] += c
    personagens = {n for n, c in canonicos.most_common(MAX_PERSONAGENS) if c >= MIN_MENCOES}
    if narrador and canonicos[narrador] > 0:
        personagens.add(narrador)
    if not personagens:
        sys.exit("Nenhum personagem encontrado.")

    mencoes_cap, arestas_cap, exemplos = contarRelacoes(
        frases_por_capitulo, textos_por_capitulo, apelidos, personagens)

    arestas_total = Counter()
    for e in arestas_cap:
        arestas_total.update(e)

    relacoes = classificarRelacoes(arestas_total, exemplos, titulo)

    grafo_layout = nx.Graph()
    grafo_layout.add_nodes_from(sorted(personagens))
    for (a, b), w in sorted(arestas_total.items()):
        grafo_layout.add_edge(a, b, weight=1 + w ** 0.3)
    for u, v, d in grafo_layout.edges(data=True):
        d["dist"] = 1 / d["weight"]
    pos = nx.kamada_kawai_layout(grafo_layout, weight="dist")

    G = nx.Graph()
    mencoes_acum = Counter()
    primeiro_cap, primeira_aresta = {}, {}
    linhas = []
    for i, ((rotulo, _), m, e) in enumerate(zip(capitulos, mencoes_cap, arestas_cap), 1):
        novos = {n for n in m if n not in G}
        for n in novos:
            G.add_node(n)
            primeiro_cap[n] = i
        mencoes_acum.update(m)
        for (a, b), w in e.items():
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
            else:
                G.add_edge(a, b, weight=w)
                primeira_aresta[tuple(sorted((a, b)))] = i
        mais_forte = e.most_common(1)[0] if e else (("-", "-"), 0)
        linhas.append({
            "capitulo": i, "titulo": rotulo,
            "num_personagens": G.number_of_nodes(),
            "num_relacoes": sum(1 for *_, d in G.edges(data=True) if d["weight"] >= MIN_ARESTA),
            "novos_personagens": ", ".join(sorted(novos)),
            "maior_relacao": f"{mais_forte[0][0]} - {mais_forte[0][1]} ({mais_forte[1]})",
        })
        desenhar(G, pos, mencoes_acum, novos, f"{titulo} — cap. {i}/{len(capitulos)}: {rotulo}",
                 pasta_frames / f"{i:03d}.png", dpi=70, relacoes=relacoes)
        print(f"Grafo: {i}/{len(capitulos)}", end="\r")
    print()

    desenhar(G, pos, mencoes_acum, set(), f"{titulo} — rede final", saida / "grafo_final.png",
             dpi=130, relacoes=relacoes)
    criarGif(pasta_frames, saida / "evolution.gif")

    with open(saida / "capitulo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    with open(saida / "relacoes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["personagem_a", "personagem_b", "peso", "primeiro_capitulo", "relacao", "motivo"])
        for a, b, d in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"]):
            par = tuple(sorted((a, b)))
            info = relacoes.get(par, {})
            w.writerow([a, b, d["weight"], primeira_aresta[par],
                        info.get("relacao", ""), info.get("motivo", "")])
    with open(saida / "personagens.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["personagem", "mencoes", "primeiro_capitulo"])
        for n, c in mencoes_acum.most_common():
            w.writerow([n, c, primeiro_cap[n]])

    print("\nPersonagems mais mencionados:", ", ".join(f"{n} ({c})" for n, c in mencoes_acum.most_common(8)))
    print("Relações mais frequentes:")
    for a, b, d in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:8]:
        rel = relacoes.get(tuple(sorted((a, b))), {}).get("relacao", "?")
        print(f"- {a} <-> {b}: {d['weight']} [{rel}]")
    print(f"\nTerminado. Resultados na pasta: {saida.resolve()}")

if __name__ == "__main__":
    main()
