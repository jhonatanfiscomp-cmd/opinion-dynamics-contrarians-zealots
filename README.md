# Robustness of consensus to contrarians and zealots under voter, majority-rule and Sznajd dynamics on complex networks

**Author:** Jhonatan Candido (Instituto de Física de São Carlos, Universidade de São Paulo)

Code and data for the manuscript (in preparation).

Simulações Monte Carlo de dinâmica de opinião binária (0/1) em redes Erdős–Rényi (ER), Barabási–Albert (BA) e Watts–Strogatz (WS), com grau médio ⟨k⟩ = 6.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `sim.py` | Simulador (numba). Quatro regras: voter, voter não-linear (α), regra da maioria (grupo de 3) e Sznajd. Contrarians e zealots (simétricos ou todos na opinião 1). 1 MCS = N atualizações. |
| `grid.py` | Define e roda os experimentos A–E (ver abaixo) e grava `results/expX.csv`. |
| `figs.py` | Lê os CSVs e gera as 5 figuras (`figs/`) e as tabelas `table_networks.csv` e `table_thresholds.csv`. |
| `make_tables.py` | Recria `results/tables.json` (Tabelas 1–4 do artigo) a partir dos CSVs. |
| `validate.py` | Compara o simulador com os tempos de consenso do TCC (N = 1000). |
| `paper/build_paper.js` | (opcional) Monta o .docx do artigo. Os caminhos dentro dele são absolutos (`/home/claude/opinion/...`); ajuste antes de usar. |
| `results/` | CSVs de saída de cada experimento e `tables.json`. |
| `figs/` | Figuras do artigo. |

## Como reproduzir

```bash
pip install -r requirements.txt
mkdir -p results figs
python grid.py A      # baseline: tempo de consenso x N                (~30 s)
python grid.py B      # contrarians: |m| estacionário x pc             (~12 min)
python grid.py C      # zealots simétricos: |m| x pz                   (~6 min)
python grid.py D      # zealots unilaterais: consenso na opinião 1     (~4 min)
python grid.py E      # contrarians: dependência com N                 (~8 min)
python figs.py        # figuras + tabelas de redes e de limiares
python make_tables.py # Tabelas 1-4 em results/tables.json
```

Tempos medidos com 2 núcleos (`Pool(2)`); veja `results/run.log` e `results/runE.log`.
Rode os scripts de dentro da pasta do projeto. As sementes são fixas, então os resultados se repetem.

## Experimentos

| Exp. | Variáveis | N | Realizações | Observável |
|---|---|---|---|---|
| A | rede × regra, sem contrarians e zealots | 250, 500, 1000, 2000 | 100 (50 em N = 2000) | tempo até consenso Tc |
| B | rede × regra × pc (0 a 0,4) | 1000 | 50 | ⟨\|m\|⟩ nos últimos 1500 de 3000 MCS |
| C | rede × regra × pz simétrico (0 a 0,3) | 1000 | 50 | ⟨\|m\|⟩ (mesma janela) |
| D | rede × regra × pz unilateral (0 a 0,1) | 1000 | 50 | probabilidade de consenso na opinião 1 (até 20.000 MCS) |
| E | rede × regra × pc, N = 250, 500, 2000 | 250–2000 | 30 | ⟨\|m\|⟩ |

Regras (coluna `model` dos CSVs): 0 = voter, 1 = voter não-linear (α = 2), 2 = regra da maioria (r = 3), 3 = Sznajd.

## Definições

- **Contrarians** (fração `pc`, fixos): quando atualizam, adotam o oposto do que a regra daria. Na regra da maioria, a opinião minoritária do grupo.
- **Zealots** (fração `pz`, fixos): nunca mudam, mas influenciam. `zmode = 0`: metade em cada opinião; `zmode = 1`: todos na opinião 1.
- **Redes**: ER com p = ⟨k⟩/(N−1); BA com m = 3; WS com k = 6 e β = 0,1. Usa-se a componente gigante e uma rede nova a cada realização.
- **Limiar** de contrarians: pc em que ⟨\|m\|\> cruza 0,5 (interpolação linear). Não é um ponto crítico.

## Observações

- Um MCS atualiza números diferentes de agentes em cada regra; compare escalas e ordenações, não tempos absolutos entre regras.
- Este simulador é independente do notebook do TCC; `validate.py` mostra que reproduz os tempos de consenso do TCC dentro da dispersão entre realizações.
- `results/expA.csv` e `expB.csv` foram gerados por uma versão anterior do `grid.py`, sem a coluna `T1_mean`. Rodar `python grid.py A` hoje produz os mesmos valores nas demais colunas (diferença máxima 0,0), e linhas de B e D conferidas também coincidem exatamente.

## Citation

If you use this code or data, please cite:

> Candido, J. *Robustness of consensus to contrarians and zealots under voter, majority-rule and Sznajd dynamics on complex networks* (manuscript in preparation).
