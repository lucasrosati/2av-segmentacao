# Definição do problema

**Problema.** Segmentação binária de pólipos em imagens de colonoscopia: para cada pixel, decidir
se ele pertence a um pólipo. A tarefa apoia a triagem do câncer colorretal.

**Dados.** Kvasir-SEG: 1000 imagens de colonoscopia com máscara do pólipo (Jha et al., 2020).

**Pergunta de pesquisa.** Com a U-Net e o transfer learning do ImageNet fixos, trocar o encoder
CNN (ResNet-34) por um encoder Transformer hierárquico (MiT-B0) muda o desempenho de segmentação?

**Hipóteses** (métrica: Dice médio no teste por run; 30 seeds pareados; bilateral,
α = 0,05):

- **H0:** μ_Dice(ResNet-34) = μ_Dice(MiT-B0)
- **H1:** μ_Dice(ResNet-34) ≠ μ_Dice(MiT-B0)
