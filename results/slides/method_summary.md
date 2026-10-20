# Resumo do método

**Problema.** Segmentação binária de pólipos em colonoscopia (Kvasir-SEG, 1000
imagens com máscara).

**Técnicas.** (A) U-Net como arquitetura de segmentação; (B) transfer learning, com o encoder
pré-treinado no ImageNet; (C) encoder CNN ResNet-34 contra (C')
encoder Transformer hierárquico MiT-B0. Só o encoder muda entre os cenários.

**Dados.** Divisão fixa, igual em todas as execuções: treino 800, validação 100,
teste 100. Imagens redimensionadas para 256×256; aumento de dados
só no treino (flip horizontal, flip vertical, rotação de 90°, cor leve).

**Treino (igual nos dois cenários).** AdamW, lr 0,0003, agenda
cosseno, perda Dice + BCE, lote 16, até 50
épocas com early stopping na Dice de validação (paciência 10).
O teste é avaliado uma única vez, com os pesos da melhor época de validação.

**Protocolo.** 30 simulações pareadas por seed (0 a 29): o mesmo
seed nos dois cenários, totalizando 60 treinos. Métrica primária: Dice médio no
teste por run.

**Hipóteses e testes.** H0: μ_Dice(ResNet-34) = μ_Dice(MiT-B0); H1: diferentes (bilateral,
α = 0,05). Shapiro-Wilk nas diferenças pareadas; t pareado se forem aproximadamente
normais, senão Wilcoxon pareado (os dois são reportados). IC 95% da diferença média por
bootstrap e tamanho de efeito d_z de Cohen.
