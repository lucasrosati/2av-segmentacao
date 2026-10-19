# splits/

Decisão recomendada D2 (Lucas pode alterar): **um split fixo** treino/val/teste 80/10/10 do Kvasir-SEG (1000 imagens), gerado de forma determinística (seed 0).

- `split_v1.json`: `{"train": [...nomes...], "val": [...], "test": [...]}` (TODO: gerar com `make splits`)
- `split_v1.sha256`: hash SHA-256 do JSON. Todo run valida o hash antes de treinar.
- O split **não muda** entre as 30 simulações; só o seed (0..29) varia, pareado entre os cenários CNN e Transformer. O teste pareado isola o efeito do encoder.
- Estes arquivos (só nomes, sem imagens) **são versionados** no git.
