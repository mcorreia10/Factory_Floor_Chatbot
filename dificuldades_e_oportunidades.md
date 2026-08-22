# The Factory Floor — Dificuldades do Projeto e Oportunidades de Melhoria

Este documento regista, de forma separada do `README.md` (estado/arquitetura) e do `CLAUDE.md`
(notas de sessão), duas coisas: as dificuldades reais encontradas ao construir este projeto, e as
oportunidades de melhoria identificadas para as próximas fases rumo ao projeto final.

## Dificuldades do projeto

### 1. Busca por embedding não é confiável para identificadores alfanuméricos exatos

Códigos de falha como `F30021` não têm "significado" próprio para um modelo de embeddings — são
etiquetas, não conceitos. Como as páginas do List Manual seguem todas a mesma estrutura repetitiva
(`Power unit: ... Cause: ... Remedy: ...`), uma pergunta genérica do tipo `"Code F30021"` ou
`"Fault code F30021, what should be checked?"` fica semanticamente parecida com **qualquer** página
de fault code, não especificamente com a certa. Testado sistematicamente neste projeto: para ~18
códigos reais, a recuperação só acertou a página certa numa fração deles, mesmo em k=15.

A mitigação encontrada (juntar o código a um sintoma real, ex. `"F30021 ground fault"`) funciona,
mas depende do operador saber ou escrever o sintoma — não resolve o caso em que alguém introduz
apenas o código, que é um caso de uso realista (ex. o código apareceu no visor do equipamento).

**Risco associado, mais grave que "não responde":** como o retriever do Chroma devolve sempre os
`k` vizinhos mais próximos (não há threshold de relevância), uma pergunta com código errado
recupera sempre conteúdo real do manual — só que de um código diferente. O prompt de sistema não
instrui o modelo a confirmar que o código pedido aparece literalmente no contexto recuperado antes
de responder, pelo que o resultado mais provável não é "documentação insuficiente", mas sim **uma
resposta bem citada e com aparência de fundamentada, sobre a falha errada**. Num contexto de
manutenção industrial, isto pode levar a aplicar o remédio errado a uma avaria real.

A solução de mercado para este tipo de problema é busca híbrida (lexical/exata + semântica),
roteamento por deteção de padrão do código, reranking, e um guardrail pós-recuperação que valide se
o código pedido aparece literalmente no contexto antes de gerar a resposta — nenhuma destas camadas
existe ainda neste projeto.

### 2. Ausência de threshold de relevância na recuperação

O retriever devolve sempre 5 chunks, mesmo para perguntas completamente fora do domínio (testado:
"o que pensas sobre futebol" ainda popula a tabela de "Sources retrieved"). O modelo recusa
responder corretamente por instrução de prompt, mas a UI mostra fontes que não são realmente
relevantes — pode confundir o operador sobre o que realmente fundamenta (ou não) a resposta.

### 3. Ausência de suíte de testes automatizada / avaliação sistemática

Toda a verificação feita até agora foi manual — correr notebooks e conduzir a aplicação ao vivo no
browser. Não existe nenhum conjunto de perguntas com resposta esperada corrida automaticamente, nem
métricas de groundedness/hallucination. Funciona como processo de QA para uma sessão de trabalho,
mas não escala nem protege contra regressões silenciosas à medida que o projeto cresce.

### 4. Histórico de manutenção ainda não alimenta o raciocínio do LLM

O histórico por máquina é hoje apenas apresentado na UI (tabela), não é passado ao modelo como
contexto adicional. Um operador a perguntar "o que fazer com esta falha" não beneficia do
conhecimento de que aquela mesma máquina já teve 3 ocorrências parecidas — essa correlação fica a
cargo do humano, não do sistema.

### 5. Extração de PDF apenas em texto — perde diagramas e chapas de identificação

O `PyPDFLoader` extrai apenas texto corrido. Diagramas de ligação elétrica, esquemas e fotos de
chapa de identificação (nameplates) — informação frequentemente crítica em manutenção industrial —
não entram no corpus pesquisável.

### 6. Estado por sessão de browser, sem identidade de operador

O `st.session_state` do Streamlit vive por sessão de browser — não há noção de "quem" está a usar a
aplicação. Duas pessoas na mesma sessão veem o mesmo estado; a mesma pessoa em duas abas tem dois
estados independentes. Não há forma de saber, a partir dos dados de hoje, quem conduziu uma
determinada pesquisa ou resolveu uma avaria. (Ver oportunidade de melhoria 1, abaixo.)

### 7. Corpus ainda desbalanceado entre famílias de equipamento

~2766 páginas de VFD vs ~913 de motor elétrico (~3:1) — já foi corrigido de um desequilíbrio inicial
de 18:1, mas ainda pode enviesar ligeiramente a qualidade de recuperação a favor de VFDs em
perguntas ambíguas sobre "General question" sem máquina selecionada.

### 8. ~~Ausência de arquitetura de agente~~ — RESOLVIDO a 2026-08-19 (mesmo dia)

Confirmado a 2026-08-19 ao ler os requisitos oficiais do projeto final: o sistema tinha de
usar uma arquitetura agêntica — um modelo que decide autonomamente que ferramentas usar e
mantém memória — não apenas uma pipeline fixa. Construído no mesmo dia:
`factory_floor/agent.py` (`langchain.agents.create_agent`, tools `search_manuals` e
`get_maintenance_history`), verificado com 4 cenários reais incluindo deteção de conflito
entre sintoma descrito e foto (o agente pede clarificação em vez de arriscar diagnóstico).
Ver `README.md`, secção "Session note (2026-08-19, cont.)", para o detalhe completo. Mantido
aqui, riscado, como registo histórico da lacuna original — não apagar o contexto.

### 9. ~~Ausência de tracing/observabilidade~~ — RESOLVIDO a 2026-08-21

Instrumentado com LangSmith: `factory_floor/tracing.py` roteia todos os runs para um
projeto dedicado `factory-floor` (sem tocar no `.env` partilhado, que aponta para o
projeto `lca-lc-foundation` de outros labs). `run_diagnostic_agent()` e `ask()` marcam
e nomeiam cada chamada e devolvem um `run_id`; `tracing.py::run_url()` constrói um
link direto para qualquer um dos dois (foi preciso um mecanismo próprio, já que
`LangChainTracer.get_run_url()` só vê runs criados pelo callback manager e falha para
chamadas decoradas com `@traceable`, como o `ask()`). Verificado ao vivo em
`notebooks/08_tracing_observability.ipynb`: um trace real multi-passo (o agente
escolheu chamar `get_maintenance_history` e depois `search_manuals`, por essa ordem,
numa pergunta real sobre F30059 na VFD-06), um trace do baseline (single-chain, para
contraste), e o trace de raciocínio sem nenhuma chamada a ferramentas do cenário de
sinais conflituosos. Ver a nota de 2026-08-21 em `CLAUDE.md` para os detalhes técnicos
(cache do SDK, ordem de import vs. `load_dotenv()`, etc.). Mantido aqui, riscado, como
registo histórico da lacuna original.

### 10. ~~Avaliação nunca corrida de forma automatizada, sem comparação com baseline~~ — RESOLVIDO a 2026-08-21

`eval_scenarios.csv` — 37 cenários de troubleshooting (≥30 exigidos), cada um com uma
causa-raiz e uma frase de evidência do manual verificadas individualmente contra o
vectorstore real (a evidência nunca repete o código/sintoma já presente na pergunta,
para não permitir pontuar por simples eco da pergunta). `factory_floor/evaluation.py`
pontua `rag.ask()` (baseline) e `run_diagnostic_agent()` (agente) de forma idêntica.
Números reais medidos em `notebooks/10_evaluation_baseline.ipynb`:
- Accuracy de causa-raiz: 78.4% agente vs. 75.7% baseline. Accuracy de evidência: 45.9%
  vs. 48.6% (próximos — eco honesto da dificuldade #1 abaixo).
- Consciência de histórico de máquina: baseline 0% (estrutural, confirmado por
  `assert`), agente citou corretamente a data real do histórico (2022-06-25).
- Resolução de conflito (foto `structural_damage` + texto a minimizar como "só um
  risco cosmético"): o agente pediu clarificação; o baseline, sem qualquer mecanismo
  de `vision_context`, aceitou a descrição ao pé da letra e concluiu "provavelmente
  seguro continuar a funcionar" — um achado real e um pouco preocupante sobre a
  diferença entre as duas arquiteturas.
- Auditoria de segurança (ver item 11 abaixo): 12/30 falhas no agente (40.0%) vs.
  32/34 no baseline (94.1%) — reportado tal como saiu, não afinado até parecer melhor.

O classificador de defeitos visuais (item 4 do checklist de conformidade no README) já
tinha esta comparação feita antes; agora o RAG/agente também tem. Mantido aqui, riscado,
como registo histórico da lacuna original.

### 11. Corpus de Fichas de Segurança (SDS) em falta — parcialmente endereçado a 2026-08-21

O spec oficial pede um corpus misto — manuais + fichas de segurança (SDS) dos
lubrificantes/óleos/produtos de limpeza do equipamento — para sustentar o "safety-first
output contract" (avisos de segurança sempre antes de qualquer passo de reparação, com
auditoria: quantas vezes as precauções não vieram primeiro).

**Atualização 2026-08-21**: a regra "safety-first" e a auditoria já existem — regra 6 em
`factory_floor/agent.py::DIAGNOSTIC_SYSTEM_PROMPT` (fundamentada nas secções de
segurança já presentes nos manuais de motor/VFD ingeridos, sem precisar de SDS) e
`factory_floor/safety.py` (juiz LLM + verificação determinística por regex,
`notebooks/09_safety_validator.ipynb`). Auditado ao nível do conjunto de avaliação
completo em `notebooks/10_evaluation_baseline.ipynb`: o agente ainda falha em 40.0%
das respostas que recomendam uma ação (12 de 30) — a regra ajuda bastante (o baseline
sem regra falha 94.1%) mas não garante 100%, mesmo confirmado ao vivo na app Streamlit
a funcionar (ver `CLAUDE.md`). **O que continua em falta**: o corpus de SDS em si — sem
ele, as precauções ficam limitadas ao que os manuais de motor/VFD já dizem sobre
segurança geral, não a fichas de segurança específicas de produtos químicos/lubrificantes.
Também continua em falta: uma versão do Safety Validator que bloqueie/avise em tempo
real na app (hoje é só uma auditoria pós-resposta usada na avaliação, decisão de âmbito
confirmada com o dono do projeto).

### 12. Dataset de defeitos ainda não é especificamente de motores/VFDs

As 4 categorias curadas do MVTec AD (`cable`, `metal_nut`, `screw`, `transistor`) foram escolhidas
por serem componentes fisicamente plausíveis num motor/VFD, mas **não são fotos reais de avarias em
motores ou variadores** — são peças genéricas de um benchmark académico de deteção de anomalias.
Isto limita quão diretamente o classificador generaliza para os defeitos reais que um operador desta
fábrica veria (desgaste de rolamentos, danos em enrolamentos, corrosão numa carcaça de motor, pó
acumulado num dissipador de VFD, etc.) — nenhum destes aparece no dataset atual. Ver oportunidade de
melhoria 3, abaixo.

### 13. Baseline zero-shot do classificador de defeitos tem desempenho pior que o acaso

Medido, não hipotético: sobre o split de teste, o modelo de visão zero-shot (`gpt-4.1-mini`) obteve
42.5% de exatidão — **pior que simplesmente prever sempre "good"** (77.1%, a classe maioritária). O
modelo zero-shot tende a inventar defeitos mesmo em fotos sem problema (recall de apenas 7% na classe
"good"). É um achado real de análise de falhas, útil para a apresentação, mas confirma que uma
abordagem zero-shot pura não é fiável para esta tarefa sem um classificador treinado a apoiar.

### 14. Recomendação de ações da foto não está fundamentada nos manuais

`recommend_actions()` gera texto de "o que fazer" a partir do conhecimento geral do LLM sobre o tipo
de defeito, não de retrieval sobre os manuais ou fichas de segurança — é avisado explicitamente na
UI ("Not grounded..."), mas o risco de um conselho genérico e subtilmente errado existe até isto ser
fundido com o RAG Tool pelo futuro Orchestrator Agent.

### 15. A pontuação da avaliação (item 10) é uma proxy por palavras-chave, não correção humana

`factory_floor/evaluation.py::keyword_score()` verifica se uma frase específica aparece literalmente
na resposta — uma resposta correta mas escrita de forma diferente conta como erro; uma resposta
errada que repita por acaso as palavras-chave conta como acerto. O threshold de 50% para "passou" é
arbitrário, escolhido uma vez e não afinado a partir dos resultados. `source_score()` verifica só a
família do manual (nome do ficheiro), não a página exata — não existe verdade fundamental (ground
truth) humana ao nível da página para este conjunto de 37 cenários. Mitigado, não eliminado: as
`expected_evidence_keywords` são sempre confirmadas ausentes da própria pergunta (para não permitir
pontuar por eco), mas continuam a ser uma proxy.

### 16. Viés de auto-preferência no juiz de segurança (item 11)

`factory_floor/safety.py::check_safety_precautions()` usa `gpt-4.1-mini` para julgar respostas
geradas pelo mesmo `gpt-4.1-mini` — um risco real de viés de auto-preferência (o modelo pode avaliar
a sua própria forma de escrever de forma mais favorável). Mitigado com um segundo verificador
determinístico por regex (`check_safety_precautions_keyword()`), sem chamada a LLM nenhuma — a taxa
de concordância entre os dois (91.9% nas respostas do agente, 78.4% nas do baseline, medida a
2026-08-21) é reportada como o grau de confiança honesto no número do juiz, não escondida.

### 17. Traces do LangSmith ficam numa conta partilhada com outros projetos não relacionados

`factory_floor/tracing.py` isola os traces deste projeto num projeto LangSmith dedicado
(`factory-floor`), mas isso é separação ao nível de *projeto*, não de conta/organização — a mesma
conta LangSmith (e a mesma `LANGSMITH_API_KEY` no `.env` partilhado) é usada por outros labs não
relacionados com este bootcamp. Os URLs de trace impressos no `notebooks/08_tracing_observability.ipynb`
(e nos seus outputs committados) contêm o UUID da organização/tenant — inofensivo para partilhar, mas
vale a pena saber que está lá antes de publicar screenshots.

## Oportunidades de melhoria

### 1. Autenticação por utilizador (user ID + password) associada a um operador específico

Hoje a aplicação não sabe quem a está a usar. A proposta é introduzir um mecanismo de login simples
que associe cada sessão a um operador identificável, e a partir daí:

- O operador responsável pela pesquisa/resolução de uma avaria fica registado nessa interação.
- O operador pode escrever, dentro da própria aplicação, os passos que efetivamente realizou para
  resolver a avaria (texto livre), com **data e hora** do registo.
- Esse registo — operador + passos + data/hora, ligado à pergunta original e à resposta
  fundamentada que o sistema deu — fica gravado no **histórico da máquina** (hoje
  `maintenance_history.csv`, estático e gerado uma vez; passaria a ser uma tabela viva, com
  escrita, não só leitura).

**Considerações técnicas a resolver no desenho:**
- `machines.py` hoje só tem funções de leitura (`load_machines`, `load_maintenance_history`,
  `get_machine_history`) — é preciso um caminho de escrita novo (ex. `append_resolution_event()`),
  e decidir se isso escreve no mesmo CSV ou numa tabela separada de "eventos gerados pela
  aplicação" vs. "histórico simulado original".
- Autenticação mínima viável para demonstração: um ficheiro/tabela de credenciais local (ex. via
  `streamlit-authenticator` ou equivalente), sem necessidade de um sistema de identidade completo
  nesta fase.
- Se várias pessoas usarem a app ao mesmo tempo, escrever no mesmo CSV a partir de sessões
  Streamlit concorrentes precisa de cuidado (condições de corrida ao escrever ficheiro) — resolúvel
  de forma simples para demo, mas a apontar como limitação conhecida se não for tratado a fundo.

### 2. Botão de envio do registo para outras aplicações da fábrica (demonstração)

Um botão único, de propósito demonstrativo, que simula o envio deste registo de resolução (operador
+ passos + data/hora + avaria) para outros sistemas da fábrica — tipicamente um CMMS (Computerized
Maintenance Management System, ex. SAP PM, IBM Maximo) ou um ERP.

Para efeitos de demonstração **não precisa de integração real** — o objetivo é mostrar o ponto de
integração conceptual, não construir a integração em si:
- Ao clicar, a aplicação pode simular o envio (ex. gravar num ficheiro/log separado a simbolizar
  "saída para o CMMS", ou mostrar uma confirmação tipo toast/mensagem de sucesso).
- Fica documentado como o ponto onde, numa fase futura real, entraria uma chamada a uma API/webhook
  de um sistema externo.

Esta oportunidade depende diretamente da 1 — só faz sentido "enviar" um registo estruturado
(operador, passos, data/hora) depois de esse registo existir.

### 3. Mais dados — fotos representativas de problemas reais de motores elétricos e VFDs

O classificador de defeitos atual foi treinado sobre um dataset académico genérico (MVTec AD),
curado apenas por analogia de componente (cabo, porca, parafuso, transístor — ver dificuldade 12
acima), não sobre avarias reais deste domínio. A melhoria de maior impacto para a fiabilidade da
Vision Tool seria reunir/gerar um conjunto de fotos verdadeiramente representativas de problemas de
motores elétricos e variadores de frequência — por exemplo: rolamentos desgastados ou com corrosão,
enrolamentos queimados ou com isolamento degradado, terminais/caixas de ligação corroídas ou com
sinais de sobreaquecimento, dissipadores de VFD com acumulação de pó, ventoinhas de arrefecimento
danificadas, conectores de cabo derretidos ou oxidados. Isto tornaria a classificação e as ações
recomendadas muito mais próximas da realidade do operador do que as 4 categorias genéricas atuais.

### 4. Áudio — diagnóstico pelo som do motor/VFD (extra, não exigido para nota)

Mesmo princípio da Vision, mas para som: o operador grava/carrega o áudio do motor ou VFD (ex. um
ruído estranho), o sistema "ouve" e sugere o problema provável + verificações a fazer — ainda
**separado** do RAG e da Vision (a fusão dos três é trabalho do futuro Orchestrator Agent, mesma
fronteira já estabelecida). Investigado a 2026-08-19, adiado por agora a pedido do dono do projeto —
os 7 requisitos nucleares do bootcamp já estão satisfeitos pela Vision (áudio seria um extra além do
exigido), e o dataset disponível precisa de preparação significativa antes de qualquer treino.

**Dataset real identificado (não genérico, ao contrário do cuidado que tivemos de ter com o MVTec):**
o **UOEMD-VAFCVS** (University of Ottawa Electric Motor Dataset — Vibration and Acoustic Faults
under Constant and Variable Speed Conditions), Mendeley Data, DOI `10.17632/msxs4vj48g`. Gravado
com um motor de indução real **acionado por um VFD real** num banco de testes (SpectraQuest
Machinery Fault & Rotor Dynamics Simulator), com microfone + 3 acelerómetros + temperatura. Defeitos
reais e diretamente relevantes ao domínio deste projeto: desequilíbrio de rotor, desalinhamento,
avarias no enrolamento do estator, desequilíbrio de tensão, rotor empenado, barras do rotor
partidas, avarias em rolamentos — ao contrário das 4 categorias genéricas do MVTec, isto é
literalmente do domínio motor+VFD.

**Porque não é tão direto quanto o MVTec foi:**
- Só 128 gravações no total (8 motores × condições de falha × velocidades constante/variável),
  muito menos que as 1502 imagens usadas na Vision — vai precisar de aumento de dados (data
  augmentation) ou uma abordagem few-shot, não uma classificação supervisionada clássica direta.
- Ficheiros são CSV multi-sensor (colunas de acelerómetro + áudio + temperatura), 10s a 42kHz —
  não é áudio "pronto a usar"; é preciso extrair só o canal acústico, cortar em clips, e
  provavelmente gerar espectrogramas/MFCCs antes de qualquer classificador (biblioteca `librosa`
  ou `torchaudio`, nenhuma delas ainda no projeto).
- Etiquetagem por código (`{Letra}-{Letra}-{Número}-{Número}`, ex. `H-H` = healthy, `B-B` =
  faulty bearing) precisa de ser mapeada para rótulos legíveis, à semelhança do que já foi feito
  para o `COARSE_LABEL_MAP` da Vision.

**Quando retomar isto**: seguir o mesmo processo rigoroso usado na Vision — não assumir a estrutura
do dataset de memória, descarregar e inspecionar primeiro, confirmar o mapeamento real de labels
antes de qualquer código de treino.
