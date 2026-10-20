# Números-chave

Gerado por `make slides-assets` a partir de `results/runs.csv` e `results/tables/stats.csv`. Não recalcule nem arredonde de outro jeito.
Código das runs: a67a0b3.

## Por cenário (conjunto de teste)

**ResNet-34** (30 runs válidas)
- Dice: 0,8957 ± 0,0100 (IC 95% da média: [0,8920; 0,8995])
- IoU: 0,8281 ± 0,0136
- Parâmetros: 24,4 M no total, 21,3 M no encoder
- Latência: 9,6 ms por imagem (mediana, lote 1, Tesla T4)
- Treino: 4,7 ± 1,2 min por run; melhor época média 29,1 de 38,0 rodadas

**MiT-B0** (30 runs válidas)
- Dice: 0,9005 ± 0,0070 (IC 95% da média: [0,8979; 0,9031])
- IoU: 0,8359 ± 0,0100
- Parâmetros: 5,5 M no total, 3,3 M no encoder
- Latência: 16,1 ms por imagem (mediana, lote 1, Tesla T4)
- Treino: 4,9 ± 1,1 min por run; melhor época média 31,4 de 40,6 rodadas

## Comparação pareada (MiT-B0 − ResNet-34, 30 pares)

- Diferença média do Dice: 0,0047 (IC 95% bootstrap: [0,0002; 0,0093])
- Tamanho de efeito d_z: 0,36
- Shapiro-Wilk nas diferenças: p = 0,225
- t pareado: p = 0,058; Wilcoxon pareado: p = 0,064
- **Decisão (α = 0,05, decidida pelo t pareado): não rejeita H0**
- IoU (secundária): diferença 0,0078, p (t) = 0,027, p (Wilcoxon) = 0,040

## Referência trivial: máscara toda zero no teste (n = 100)

- Dice: 0,0000; IoU: 0,0000
