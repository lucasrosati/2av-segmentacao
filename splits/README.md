# splits/

Decisão D2, confirmada e usada nas 60 execuções: **um split fixo** treino/val/teste 80/10/10 do Kvasir-SEG (1000 imagens), gerado de forma determinística (seed 0).

- `split_v1.json`: `{"train": [...nomes...], "val": [...], "test": [...]}` (gerado com `make splits`; 800/100/100 imagens)
- `split_v1.sha256`: hash SHA-256 do JSON. Todo run valida o hash antes de treinar.
- O split **não muda** entre as 30 simulações; só o seed (0..29) varia, pareado entre os cenários CNN e Transformer. O teste pareado compara os encoders com o restante do protocolo fixo.
- Estes arquivos (só nomes, sem imagens) **são versionados** no git.
