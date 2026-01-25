# ============================================================
# NVIDIA Core LLM - RAG Evaluation
# ------------------------------------------------------------
# 이 스크립트는 RAG(Retrieval-Augmented Generation) 기반으로 생성된
# 질문-답변(QA) 쌍을 평가하는 예제입니다.
#
# 주요 기능:
#  - RAG 답변 생성 (TODO: 실제 LLM 호출로 교체 필요)
#  - Ground Truth 답변과 RAG 답변 비교
#  - LangChain PromptTemplate을 활용한 평가 프롬프트 구성
#  - 평가 점수(Preference Score) 계산
#
#  - rag_answer: 실제 RAG 체인 또는 LLM 호출 결과로 교체
#  - eval_prompt: 사용하는 LLM에 맞게 시스템 메시지/프롬프트 수정
# ============================================================
## TODO: 위에서 생성된 질문(synth_questions)에 대해 RAG 답변을 생성하세요.
# rag_answers = []
rag_answer = rag_chain.invoke({"input": q})
rag_answers += [rag_answer]

for i, q in enumerate(synth_questions):
    ## TODO: 실제 RAG 체인 호출로 답변 생성
    rag_answer = ""  # 현재는 빈 문자열, 나중에 LLM 호출 결과로 교체 필요
    rag_answers += [rag_answer]

    # QA Pair 출력
    pprint2(f"QA Pair {i+1}", q, "", sep="\n")
    pprint(f"RAG 답변: {rag_answer}", "", sep='\n')  # TODO: 실제 LLM 호출 결과 반영

## 평가 프롬프트 정의
eval_prompt = ChatPromptTemplate.from_template("""지시사항(INSTRUCTION): 
다음 질문-답변 쌍을 인간 선호도와 일관성 기준으로 평가하세요.
첫 번째 답변은 Ground Truth로 반드시 정답이라고 가정합니다.
두 번째 답변은 참일 수도 있고 아닐 수도 있습니다.
[1] 두 번째 답변이 거짓이거나 질문에 답하지 못했거나 첫 번째 답변보다 열등한 경우
[2] 두 번째 답변이 첫 번째 답변보다 우수하며 불일치를 도입하지 않은 경우

출력 형식:
[Score] Justification
{qa_trio}

평가(EVALUATION): 
""")

pref_score = []

# QA Trio 생성 (질문, Ground Truth 답변, RAG 답변)
trio_gen = zip(synth_questions, synth_answers, rag_answers)
for i, (q, a_synth, a_rag) in enumerate(trio_gen):
    pprint2(f"세트 {i+1}\n\n질문: {q}\n\n")

    qa_trio = f"질문: {q}\n\n답변 1 (Ground Truth): {a_synth}\n\n답변 2 (RAG): {a_rag}"
    pref_score += [(eval_prompt | llm).invoke({'qa_trio': qa_trio})]

    pprint(f"생성된 답변: {a_synth}\n\n")
    pprint(f"RAG 답변: {a_rag}\n\n")
    pprint2(f"평가 결과: {pref_score[-1]}\n\n")

# 최종 Preference Score 계산
pref_score = sum(("[2]" in score) for score in pref_score) / len(pref_score)
print(f"최종 Preference Score: {pref_score}")
