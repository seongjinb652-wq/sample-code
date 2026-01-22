# sample-code
sample-code
이 코드는 LangGraph를 활용한 기업 분석 자동화 시스템입니다. 주요 특징과 구조를 분석해드리겠습니다.
📋 시스템 개요
이 코드는 국내 상장 기업의 투자 분석 리포트를 자동으로 생성하는 에이전트 시스템입니다. LangGraph의 상태 기반 워크플로우를 사용하여 데이터 수집부터 보고서 작성까지 자동화합니다.
🏗️ 핵심 아키텍처
1. 상태 관리 (CompanyState)
pythonclass CompanyState(TypedDict):
    question: str                      # 사용자 질문
    company_hint: str                  # 회사명/티커 힌트
    ticker: str                        # 확인된 티커
    company_name: str                  # 회사명
    price_df: pd.DataFrame            # 가격 데이터
    market_price_snapshot: dict       # 시장 스냅샷
    news: List[dict]                  # 뉴스 데이터
    notes: List[str]                  # 진행 단계 기록
    analysis_draft: str               # 초안 보고서
    final_report: str                 # 최종 보고서
```

### 2. **워크플로우 노드 (6단계)**
```
시작 → Decision_Company → Get_MarketPrice → Get_NewsData 
     → Draft_Report → Final_Report → Save_Report → 종료
각 노드의 역할:

Decision_Company: DART API + 네이버 금융으로 티커 추출
Get_MarketPrice: yfinance로 가격/밸류에이션 데이터 수집
Get_NewsData: 네이버 뉴스 크롤링 + reranker로 관련성 검증
Draft_Report: LLM으로 분석 초안 작성
Final_Report: 최종 보고서 구조화
Save_Report: Word 문서(.docx) 저장

🔍 주요 기능 분석
티커 매핑 시스템 (강점)
python# DART 기업 리스트 → rapidfuzz 유사도 매칭 → 네이버 금융으로 시장 구분
# KOSPI(.KS) / KOSDAQ(.KQ) 자동 구분

한글 회사명으로 정확한 티커 추출
유사도 기반 검색 (cutoff=75)

뉴스 수집 + 관련성 검증
python# 네이버 뉴스 크롤링 → BGE reranker로 관련성 점수 계산
# 상위 N개 선별 (기본 5개)

Reranker를 통해 질 높은 뉴스만 선별
본문 추출 및 요약 기능

LLM 기반 보고서 생성
python# GPT-4o-mini 또는 Ollama 모델 선택 가능
# 구조화된 프롬프트로 5개 섹션 자동 생성

회사 개요, 최근 동향, 밸류에이션, 리스크, 결론
보수적 코멘트 및 투자 권유 배제

💪 강점

엔드투엔드 자동화: 입력(회사명) → 출력(Word 보고서)까지 완전 자동
국내 시장 특화: DART, 네이버 금융 등 한국 데이터 소스 활용
관련성 검증: Reranker로 노이즈 필터링
확장 가능한 구조: LangGraph 기반으로 노드 추가 용이
멀티 LLM 지원: OpenAI/Ollama 선택 가능

⚠️ 개선 포인트
1. 에러 핸들링 부족
python# 현재 코드
info = yf.Ticker(ticker_name).get_info() or {}

# 개선안
try:
    info = yf.Ticker(ticker_name).get_info()
except Exception as e:
    add_note(state, f"[Error] 티커 정보 획득 실패: {e}")
    info = {}
2. 뉴스 소스 의존성

네이버 뉴스 크롤링 실패 시 대체 수단 부재
Tavily API 폴백이 있지만 한국어 뉴스 품질 불확실

3. 밸류에이션 한계
python# 단순 PE, PB, 52주 고저만 활용
# 개선: ROE, EV/EBITDA, 동종업계 비교 추가
4. LLM 프롬프트 고도화 필요
python# 현재: 단일 프롬프트로 전체 보고서 생성
# 개선: 섹션별 프롬프트 분리 + Chain-of-Thought
5. 비동기 처리 부재
python# 뉴스 크롤링/LLM 호출을 순차 처리
# 개선: async/await로 병렬 처리 → 속도 향상
🚀 확장 아이디어

재무제표 분석 노드 추가

DART에서 사업보고서 추출
매출/영업이익/순이익 추세 분석


경쟁사 비교 기능

동종업계 기업 자동 추출
밸류에이션/성장성 비교 테이블


실시간 대시보드

Streamlit/Gradio로 웹 UI 구축
그래프 시각화 추가


다국어 지원

해외 상장 기업도 분석 가능하도록 확장



📊 코드 품질 평가
항목평가코멘트구조⭐⭐⭐⭐☆LangGraph 활용 우수, 모듈화 가능안정성⭐⭐⭐☆☆에러 핸들링 보완 필요확장성⭐⭐⭐⭐☆노드 추가 용이성능⭐⭐⭐☆☆비동기 처리로 개선 가능실용성⭐⭐⭐⭐⭐실제 업무 적용 가능
🎯 결론
이 코드는 실무에 즉시 적용 가능한 수준의 기업 분석 자동화 시스템입니다. 특히 국내 상장사 분석에 특화되어 있으며, LangGraph의 상태 기반 워크플로우를 잘 활용했습니다.
에러 핸들링과 성능 최적화를 보완하면 프로덕션 레벨로 발전시킬 수 있는 훌륭한 기반 코드입니다.
추가로 궁금하신 부분이나 특정 기능에 대한 상세 분석이 필요하시면 말씀해주세요!
