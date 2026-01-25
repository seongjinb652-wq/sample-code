# ============================================================
# NVIDIA Core LLM - Gradio Chatbot Interface
# ------------------------------------------------------------
# 이 스크립트는 LangChain과 Gradio를 활용하여 문서 기반 챗봇을
# 웹 인터페이스로 실행하는 예제입니다.
#
# 주요 기능:
#  - 초기 메시지를 포함한 Gradio Chatbot UI 생성
#  - chat_gen 함수를 통한 대화 스트리밍 처리
#  - 예외 발생 시 안전하게 서버 종료
# ✅ 추천 버전  (2016.01월 기준)
# Python 3.10 → 가장 안정적이고, 대부분의 라이브러리와 호환성이 뛰어남
# Python 3.11 → 성능 최적화가 잘 되어 있고, 최신 기능을 활용 가능
# Python 3.12 → 최신 버전이지만 일부 라이브러리 호환성 문제가 있을 수 있음
# ============================================================
import gradio as gr

# 초기 챗봇 메시지
chatbot = gr.Chatbot(value=[[None, initial_msg]])

# Gradio 인터페이스 정의
demo = gr.ChatInterface(
    fn=chat_gen,          # 대화 생성 함수 연결
    chatbot=chatbot       # 챗봇 UI 연결
).queue()

# Gradio 실행 및 예외 처리
try:
    demo.launch(debug=True, share=True, show_api=False)  # 디버그 모드, 공유 URL 생성
    demo.close()  # 실행 종료 후 서버 닫기
except Exception as e:
    demo.close()  # 오류 발생 시 서버 닫기
    print("에러 발생:", e)
    raise e
