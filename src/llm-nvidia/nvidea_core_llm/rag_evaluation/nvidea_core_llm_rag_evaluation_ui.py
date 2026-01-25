# ============================================================ 
# NVIDIA Core LLM - RAG Evaluation with Gradio UI 
# ------------------------------------------------------------ 
# 이 스크립트는 RAG 기반 QA 평가를 Gradio 인터페이스로 연결하여 
# 웹 브라우저에서 직접 질문-답변 쌍을 평가할 수 있도록 합니다. 
# 
# 주요 기능: 
# - RAG 답변 생성 (TODO: 실제 RAG 체인 연결 필요) 
# - Ground Truth 답변과 RAG 답변 비교 
# - LangChain PromptTemplate을 통한 평가 프롬프트 구성 
# - Gradio UI를 통해 질문/답변/평가 결과 시각화 
# ======================================================

import gradio as gr 
from langchain_core.prompts import ChatPromptTemplate 
# TODO: 실제 synth_questions, synth_answers, rag_chain 연결 필요 
synth_questions = ["예시 질문 1", "예시 질문 2"] 
synth_answers = ["예시 정답 1", "예시 정답 2"] 
rag_answers = ["예시 RAG 답변 1", "예시 RAG 답변 2"] 
# 평가 프롬프트 정의 
eval_prompt = ChatPromptTemplate.from_template("""지시사항(INSTRUCTION): 
다음 질문-답변 쌍을 인간 선호도와 일관성 기준으로 평가하세요. 
첫 번째 답변은 Ground Truth로 반드시 정답이라고 가정합니다. 
두 번째 답변은 참일 수도 있고 아닐 수도 있습니다. 
[1] 두 번째 답변이 거짓이거나 질문에 답하지 못했거나 첫 번째 답변보다 열등한 경우 
[2] 두 번째 답변이 첫 번째 답변보다 우수하며 불일치를 도입하지 않은 경우 

출력 형식: 
[Score] Justification 
{qa_trio} 평가(EVALUATION): 
""") 
# 평가 함수 
def evaluate(index):
  q = synth_questions[index]
  a_synth = synth_answers[index]
  a_rag = rag_answers[index]
  
  qa_trio = f"질문: {q}\n\n답변 1 (Ground Truth): {a_synth}\n\n답변 2 (RAG): {a_rag}" 
  # TODO: 실제 LLM 호출로 교체 필요
  evaluation = f"[1] Justification: (예시 평가 결과)\n
  질문: {q}\n정답: {a_synth}\nRAG: {a_rag}" 
    
  return q, a_synth, a_rag, evaluation 
  
# Gradio UI 구성 
with gr.Blocks() as demo:
  gr.Markdown("## 📊 RAG QA 평가 인터페이스")
  
  index_input = gr.Number(label="QA Pair Index (0부터 시작)", value=0)
  question_output = gr.Textbox(label="질문")
  synth_output = gr.Textbox(label="Ground Truth 답변")
  rag_output = gr.Textbox(label="RAG 답변")
  eval_output = gr.Textbox(label="평가 결과")
  eval_button = gr.Button("평가 실행")
  eval_button.click(
    evaluate,
    inputs=index_input,
    outputs=[question_output, synth_output, rag_output, eval_output]
  )
  
demo.launch()
